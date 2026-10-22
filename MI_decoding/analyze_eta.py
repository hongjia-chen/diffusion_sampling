#!/usr/bin/env python3
"""Analysis of a measure_eta.py run (ETA_STUDY.md §5). CPU only.

    python analyze_eta.py --output-root outputs/eta/heldout_b16

Reads <output-root>/<dataset>/{steps,tasks}.jsonl for every dataset directory
and writes <output-root>/analysis/{report.md,summary.json}. The unit of
analysis is a (measured step, rank j ≥ 2) pair; rank 1 conditions on nothing.
Slots after a committed stop token, or after a stop token placed in the same
probe row, are padding and are left out of the per-pair statistics (the
prefix replays keep them: the decoder faces them too). Uncertainty is a
bootstrap over tasks, since steps of one task are correlated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

EPSILONS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
QUANTILES = (0.5, 0.9, 0.99, 0.999)
BIN_EDGES = (0.0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0 + 1e-9)
CONTEXT_EDGES = (0, 256, 1024, 2048, 1 << 30)
DIST_EDGES = (1, 2, 3, 5, 1 << 30)
CONF_EDGES = (0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1.0 + 1e-9)
FIELDS = ("D_P", "D_V", "eta", "tv_PV", "gap_P", "conf_P", "conf_V", "lr_P", "lr_V")


def stream(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stop_ids(dataset_dir: Path) -> set[int]:
    for path in sorted(dataset_dir.glob("run_metadata_*.json")):
        return set(json.loads(path.read_text())["stop_token_ids"])
    raise FileNotFoundError(f"no run_metadata in {dataset_dir}")


def padding_mask(step: dict, stops: set[int]) -> list[bool]:
    """Per rank: True when the slot lies after a stop token (committed, or placed in its row)."""
    first = step["first_stop_slot"]
    slots, tokens = step["slots"], step["tokens"]
    out = []
    for j, slot in enumerate(slots):
        after_committed = first >= 0 and slot > first
        after_placed = any(tokens[l] in stops and slots[l] < slot for l in range(j))
        out.append(after_committed or after_placed)
    return out


def load(dataset_dir: Path) -> dict:
    stops = stop_ids(dataset_dir)
    tasks = {row["example_id"]: row for row in stream(dataset_dir / "tasks.jsonl")}
    steps = [row for row in stream(dataset_dir / "steps.jsonl") if row["example_id"] in tasks]
    columns: dict[str, list] = {name: [] for name in FIELDS}
    extra = {name: [] for name in ("task", "rank", "k", "fill", "context", "dist", "flip_P", "flip_V",
                                   "agree_V", "eta_self", "noise_q", "step_index")}
    for index, step in enumerate(steps):
        padding = padding_mask(step, stops)
        for j in range(1, step["k"]):
            if padding[j]:
                continue
            for name in FIELDS:
                columns[name].append(step[name][j])
            extra["task"].append(step["example_id"])
            extra["rank"].append(j + 1)
            extra["k"].append(step["k"])
            extra["fill"].append(step["filled_in_block"])
            extra["context"].append(step["context_tokens"])
            extra["dist"].append(step["dist_new"][j])
            extra["flip_P"].append(step["argmax_pP"][j] != step["argmax_qP"][j])
            extra["flip_V"].append(step["argmax_pV"][j] != step["argmax_qV"][j])
            extra["agree_V"].append(step["argmax_qV"][j] == step["tokens"][j])
            extra["eta_self"].append(step["eta_self"][j] if "eta_self" in step else math.nan)
            extra["noise_q"].append(step["noise_q"][j] if "noise_q" in step else math.nan)
            extra["step_index"].append(index)
    pairs = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    for name, values in extra.items():
        pairs[name] = np.asarray(values, dtype=object if name == "task" else float)
    pairs["gap_plus"] = np.maximum(pairs["D_P"] - pairs["D_V"], 0.0)
    return {"name": dataset_dir.name, "tasks": tasks, "steps": steps, "pairs": pairs, "stops": stops}


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------

def quantiles(values: np.ndarray, qs=QUANTILES) -> dict[str, float]:
    values = values[~np.isnan(values)]
    if not values.size:
        return {f"q{q:g}": math.nan for q in qs} | {"max": math.nan, "mean": math.nan, "n": 0}
    out = {f"q{q:g}": float(np.quantile(values, q)) for q in qs}
    return out | {"max": float(values.max()), "mean": float(values.mean()), "n": int(values.size)}


def task_groups(tasks: np.ndarray) -> list[np.ndarray]:
    order = {}
    for i, task in enumerate(tasks):
        order.setdefault(task, []).append(i)
    return [np.asarray(v) for v in order.values()]


def bootstrap_quantile(values: np.ndarray, groups: list[np.ndarray], q: float,
                       draws: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Pooled q-quantile and its 95% interval, resampling whole tasks."""
    rng = np.random.default_rng(seed)
    estimate = float(np.quantile(values, q))
    samples = []
    for _ in range(draws):
        picked = rng.integers(0, len(groups), len(groups))
        samples.append(np.quantile(values[np.concatenate([groups[i] for i in picked])], q))
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    label = label.astype(bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if not positives or not negatives:
        return math.nan
    ranks = rankdata(score)
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def binned(pairs: dict, by: str, edges=BIN_EDGES) -> list[dict]:
    out = []
    key = pairs[by]
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (key >= low) & (key < high)
        if not inside.any():
            continue
        eta, d_p = pairs["eta"][inside], pairs["D_P"][inside]
        out.append({
            "bin": f"[{low:g}, {min(high, 1.0):g})", "n": int(inside.sum()),
            f"{by}_median": float(np.median(key[inside])),
            "eta_q50": float(np.quantile(eta, 0.5)), "eta_q90": float(np.quantile(eta, 0.9)),
            "eta_q99": float(np.quantile(eta, 0.99)),
            "D_P_q99": float(np.quantile(d_p, 0.99)),
            "gap_plus_q99": float(np.quantile(pairs["gap_plus"][inside], 0.99)),
            "eta_over_D_P": float(eta.sum() / d_p.sum()) if d_p.sum() > 0 else math.nan,
            "flip_P": float(pairs["flip_P"][inside].mean()),
        })
    return out


def by_covariate(pairs: dict, name: str, edges) -> list[dict]:
    out = []
    key = pairs[name]
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (key >= low) & (key < high)
        if inside.any():
            out.append({"bin": f"[{low}, {high if high < 1 << 29 else '∞'})", "n": int(inside.sum()),
                        **{f"eta_{k}": v for k, v in quantiles(pairs["eta"][inside], (0.5, 0.99)).items()
                           if k in ("q0.5", "q0.99")},
                        "D_P_q99": float(np.quantile(pairs["D_P"][inside], 0.99))})
    return out


def false_accepts(pairs: dict, kappas: dict[str, float]) -> list[dict]:
    """Among slots the verifier calls quiet (D^V ≤ ε): how often is D^P > ε + κ?"""
    out = []
    for epsilon in EPSILONS:
        quiet = pairs["D_V"] <= epsilon
        row = {"epsilon": epsilon, "n_quiet": int(quiet.sum()),
               "share_quiet": float(quiet.mean()) if quiet.size else math.nan,
               "flip_P_in_quiet": float(pairs["flip_P"][quiet].mean()) if quiet.any() else math.nan}
        for label, kappa in kappas.items():
            row[f"D_P>eps+{label}"] = float((pairs["D_P"][quiet] > epsilon + kappa).mean()) if quiet.any() else math.nan
        out.append(row)
    return out


def prefix_replay(steps: list[dict], rule, label: str) -> dict:
    """Replay a commit rule on every measured step: the longest prefix whose ranks 2..m pass."""
    accepted, flips, slots, large = [], 0, 0, 0
    for step in steps:
        m = 1
        while m < step["k"] and rule(step, m):
            m += 1
        accepted.append(m)
        for j in range(1, m):
            slots += 1
            flips += step["argmax_pP"][j] != step["argmax_qP"][j]
    return {"rule": label, "steps": len(steps), "mean_accepted": float(np.mean(accepted)) if accepted else math.nan,
            "accepted_slots": slots, "flip_P_rate": flips / slots if slots else math.nan}


def cumulative(steps: list[dict], max_rank: int = 16) -> list[dict]:
    """Mean over steps of Σ_{j ≤ m} η_j (and of D^P, D^V), for steps with k ≥ m."""
    out = []
    for m in range(2, max_rank + 1):
        rows = [s for s in steps if s["k"] >= m]
        if not rows:
            break
        out.append({"m": m, "steps": len(rows),
                    "sum_eta": float(np.mean([sum(s["eta"][:m]) for s in rows])),
                    "sum_D_P": float(np.mean([sum(s["D_P"][:m]) for s in rows])),
                    "sum_D_V": float(np.mean([sum(s["D_V"][:m]) for s in rows]))})
    return out


def split_half(pairs: dict, q: float = 0.99) -> dict:
    """κ = q-quantile of η on half the tasks; coverage P(η ≤ κ) on the other half."""
    half = np.array([int(hashlib.sha256(str(t).encode()).hexdigest(), 16) % 2 for t in pairs["task"]])
    if not ((half == 0).any() and (half == 1).any()):
        return {}
    out = {}
    for fit, test in ((0, 1), (1, 0)):
        kappa = float(np.quantile(pairs["eta"][half == fit], q))
        out[f"fit_half_{fit}"] = {"kappa": kappa, "coverage_other_half": float((pairs["eta"][half == test] <= kappa).mean()),
                                  "n_fit": int((half == fit).sum()), "n_test": int((half == test).sum())}
    return out


def affine_envelope(table: list[dict]) -> dict:
    """Least-squares line through the per-D^V-bin q99 of η (bins with ≥ 50 pairs)."""
    usable = [row for row in table if row["n"] >= 50]
    if len(usable) < 2:
        return {"slope": math.nan, "intercept": math.nan, "bins": len(usable)}
    x = np.array([row["D_V_median"] for row in usable])
    y = np.array([row["eta_q99"] for row in usable])
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "bins": len(usable)}


def analyze(data: dict) -> dict:
    pairs, steps, tasks = data["pairs"], data["steps"], data["tasks"]
    groups = task_groups(pairs["task"])
    kappa = {}
    for q in (0.9, 0.99, 0.999):
        for name in ("eta", "gap_plus"):
            estimate, low, high = bootstrap_quantile(pairs[name], groups, q)
            kappa[f"{name}_q{q:g}"] = {"value": estimate, "ci95": [low, high]}
    kappas = {"0": 0.0, "eta_q0.9": kappa["eta_q0.9"]["value"], "eta_q0.99": kappa["eta_q0.99"]["value"]}
    by_dv = binned(pairs, "D_V")
    replays = []
    for epsilon in EPSILONS:
        replays.append(prefix_replay(steps, lambda s, m, e=epsilon: s["D_V"][m] <= e, f"D_V ≤ {epsilon:g}"))
    for label, value in (("0", 0.0), ("eta_q0.99", kappas["eta_q0.99"])):
        replays.append(prefix_replay(
            steps, lambda s, m, v=value: s["D_V"][m] + v < s["gap_P"][m] / 2, f"D_V + κ({label}) < gap_P/2"))
    floor = pairs["eta_self"][~np.isnan(pairs["eta_self"])]
    return {
        "counts": {
            "tasks": len(tasks), "runaway_tasks": sum(t["maximum_length_termination"] for t in tasks.values()),
            "measured_steps": len(steps), "pairs": int(pairs["eta"].size),
            "mean_output_tokens": float(np.mean([t["output_tokens"] for t in tasks.values()])) if tasks else math.nan,
            "mean_measure_ms": float(np.mean([s["measure_ms"] for s in steps])) if steps else math.nan,
            "baseline_matches": [t.get("matches_baseline") for t in tasks.values() if "matches_baseline" in t],
        },
        "floor": {"eta_self": quantiles(floor), "noise_q": quantiles(pairs["noise_q"][~np.isnan(pairs["noise_q"])]),
                  "eta_on_floor_tasks": quantiles(pairs["eta"][~np.isnan(pairs["eta_self"])])},
        "distributions": {name: quantiles(pairs[name]) for name in ("D_P", "D_V", "eta", "gap_plus", "tv_PV")},
        "share": {"D_P>1e-3": float((pairs["D_P"] > 1e-3).mean()), "D_P>0.1": float((pairs["D_P"] > 0.1).mean()),
                  "flip_P": float(pairs["flip_P"].mean()), "flip_V": float(pairs["flip_V"].mean()),
                  "agree_V": float(pairs["agree_V"].mean())},
        "kappa": kappa,
        "by_D_V": by_dv, "by_D_P": binned(pairs, "D_P"), "by_conf_P": binned(pairs, "conf_P", CONF_EDGES),
        "affine_envelope_q99": affine_envelope(by_dv),
        "false_accepts": false_accepts(pairs, kappas),
        "prefix_replay": replays,
        "cumulative": cumulative(steps),
        "auroc": {"D_V→flip_P": auroc(pairs["D_V"], pairs["flip_P"]),
                  "|lr_V|→flip_P": auroc(np.abs(pairs["lr_V"]), pairs["flip_P"]),
                  "D_V→D_P>0.1": auroc(pairs["D_V"], pairs["D_P"] > 0.1)},
        "split_half_q0.99": split_half(pairs) if pairs["eta"].size else {},
        "by_rank": by_covariate(pairs, "rank", tuple(range(2, 18))),
        "by_fill": by_covariate(pairs, "fill", (0, 1, 4, 8, 12, 16, 32, 64)),
        "by_context": by_covariate(pairs, "context", CONTEXT_EDGES),
        "by_dist_new": by_covariate(pairs, "dist", DIST_EDGES),
    }


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------

def fmt(value) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "–"
        return f"{value:.3g}"
    return str(value)


def table(rows: list[dict]) -> str:
    if not rows:
        return "(none)\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(row[k]) for k in keys) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def render(results: dict[str, dict]) -> str:
    out = ["# η study: analysis\n"]
    out.append("## Overview\n")
    rows = []
    for name, r in results.items():
        c, d = r["counts"], r["distributions"]
        matches = c["baseline_matches"]
        rows.append({"dataset": name, "tasks": c["tasks"], "runaway": c["runaway_tasks"], "steps": c["measured_steps"],
                     "pairs": c["pairs"], "η q50": d["eta"]["q0.5"], "η q99": d["eta"]["q0.99"],
                     "η q99.9": d["eta"]["q0.999"], "D_P q99": d["D_P"]["q0.99"],
                     "(D_P−D_V)+ q99": d["gap_plus"]["q0.99"], "floor η_self q99": r["floor"]["eta_self"]["q0.99"],
                     "baseline match": f"{sum(bool(m) for m in matches)}/{len(matches)}" if matches else "–"})
    out.append(table(rows))
    for name, r in results.items():
        out.append(f"\n## {name}\n")
        out.append(f"Shares: {', '.join(f'{k} {fmt(v)}' for k, v in r['share'].items())}. "
                   f"AUROC: {', '.join(f'{k} {fmt(v)}' for k, v in r['auroc'].items())}. "
                   f"Mean probe time {fmt(r['counts']['mean_measure_ms'])} ms/step.\n")
        out.append("\n**κ candidates** (pooled quantile, 95% CI by task bootstrap)\n\n")
        out.append(table([{"quantity": k, "value": v["value"], "ci_low": v["ci95"][0], "ci_high": v["ci95"][1]}
                          for k, v in r["kappa"].items()]))
        split = r["split_half_q0.99"]
        if split:
            out.append("\nSplit half (κ = q99 of η on one half, coverage on the other): " + "; ".join(
                f"{k}: κ {fmt(v['kappa'])}, coverage {fmt(v['coverage_other_half'])}" for k, v in split.items()) + "\n")
        env = r["affine_envelope_q99"]
        out.append(f"\nAffine envelope of η q99 vs D^V (bins ≥ 50 pairs): slope {fmt(env['slope'])}, "
                   f"intercept {fmt(env['intercept'])} over {env['bins']} bins.\n")
        out.append("\n**η binned by D^V**\n\n" + table(r["by_D_V"]))
        out.append("\n**η binned by D^P**\n\n" + table(r["by_D_P"]))
        out.append("\n**η binned by the 8B's confidence in y_j** (pilot 2026-10-18: large η sits at low confidence)\n\n"
                   + table(r["by_conf_P"]))
        out.append("\n**False accepts per slot** (among D^V ≤ ε)\n\n" + table(r["false_accepts"]))
        out.append("\n**Prefix replay** (commit ranks 1..m while ranks 2..m pass; flip = 8B argmax changes)\n\n"
                   + table(r["prefix_replay"]))
        out.append("\n**Cumulative over ranks** (mean over steps with k ≥ m)\n\n" + table(r["cumulative"]))
        out.append("\n**By rank**\n\n" + table(r["by_rank"]))
        out.append("\n**By fill level**\n\n" + table(r["by_fill"]))
        out.append("\n**By context length**\n\n" + table(r["by_context"]))
        out.append("\n**By distance to the nearest newly filled slot**\n\n" + table(r["by_dist_new"]))
        out.append("\n**Numeric floor** (first tasks only): η_self " + ", ".join(
            f"{k} {fmt(v)}" for k, v in r["floor"]["eta_self"].items()) + "; η on the same pairs " + ", ".join(
            f"{k} {fmt(v)}" for k, v in r["floor"]["eta_on_floor_tasks"].items()) + "\n")
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", help="default: every subdirectory with a steps.jsonl")
    args = parser.parse_args()
    names = args.datasets or sorted(p.parent.name for p in args.output_root.glob("*/steps.jsonl"))
    loaded = {name: load(args.output_root / name) for name in names}
    if len(loaded) > 1:
        pooled_pairs = {key: np.concatenate([d["pairs"][key] for d in loaded.values()]) for key in next(iter(loaded.values()))["pairs"]}
        loaded["all"] = {"name": "all", "pairs": pooled_pairs,
                         "steps": [s for d in list(loaded.values()) for s in d["steps"]],
                         "tasks": {f"{d['name']}/{k}": v for d in list(loaded.values()) for k, v in d["tasks"].items()}}
    results = {name: analyze(data) for name, data in loaded.items()}
    target = args.output_root / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(results, indent=1, default=float))
    report = render(results)
    (target / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
