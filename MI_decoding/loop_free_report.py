#!/usr/bin/env python3
"""Loop-free throughput for the EC2 block-16 runs: outputs/loop_free/ec2_b16.{csv,md}.

A capped task ran to the 4096-token canvas without a stop token, usually a
repetition loop (`maximum_length_termination`). Under the parallel decoders
(dynamic θ, the MI rule) those are the cheapest tokens there are, so they lift
tokens/s and the speedup over fixed:1, and they are graded wrong. This report
gives each cell's tokens/s three ways, each with its ratio to fixed:1:

  raw        every task of the set C (below), runaways included; the
             summary's `tokens_per_second` when C is the whole cell
  loop-free  C less the cell's own capped tasks (fixed:1 likewise less its own)
  paired     P = C less every task capped in this method, fixed:1, dynamic:0.95
             or dynamic:0.90; fixed:1 and both dynamic arms are re-aggregated on
             exactly P, so the ratios compare the same tasks

C is the tasks of the cell that the baseline's fixed:1 and dynamic arms also ran
(dev250 MI roots: their 250 GSM8K / MATH tasks). Every rate is the corpus rate,
sum(output tokens) / sum(wall seconds), prefill included, as in
summarize_results.py. Scores on C and on P, and how many capped tasks scored
correct, show what leaving the runaways out does to quality.

Baseline: outputs/opencompass/SDAR-8B-Chat-b16-ec2 (fixed:1/2/4/8, dynamic
0.95/0.90). MI: every outputs/mi/8b_proposes_b16_ec2* root. EC2 A10G timing:
compare only with each other. Reads records(.scored).jsonl only.

    python loop_free_report.py
"""
from __future__ import annotations

import csv
import datetime
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT.parent / "harness" / "scripts"))
from summarize_results import capped, corpus_rates  # noqa: E402

OUTPUTS = PROJECT / "outputs"
BASELINE = OUTPUTS / "opencompass/SDAR-8B-Chat-b16-ec2"
MI_ROOTS = sorted((OUTPUTS / "mi").glob("8b_proposes_b16_ec2*"))
OUT_DIR = OUTPUTS / "loop_free"
BENCHMARKS = ["gsm8k", "math_500", "humaneval", "mbpp_sanitized"]
BENCH_LABEL = {"gsm8k": "GSM8K", "math_500": "MATH-500", "humaneval": "HumanEval", "mbpp_sanitized": "MBPP (sanitized)"}
BASE_METHODS = ["fixed_k_1", "fixed_k_2", "fixed_k_4", "fixed_k_8", "dynamic_theta_0p95", "dynamic_theta_0p90"]
FIXED1, DYN95, DYN90 = "fixed_k_1", "dynamic_theta_0p95", "dynamic_theta_0p90"
ANCHORS = (FIXED1, DYN95, DYN90)


def records(cell: Path) -> dict[str, dict]:
    """A cell's records by example_id, graded ones when they exist (as summarize_results.py)."""
    path = cell / "records.scored.jsonl"
    if not path.exists():
        path = cell / "records.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    unscored = [r["example_id"] for r in rows if r.get("score") is None]
    if unscored:
        raise SystemExit(f"{path}: {len(unscored)} tasks have no score (run score_opencompass.py)")
    out = {str(r["example_id"]): r for r in rows}
    if len(out) != len(rows):
        raise SystemExit(f"{path}: duplicate example_ids")
    return out


def label(method: str) -> str:
    if method.startswith("fixed_k_"):
        return f"fixed:{method.removeprefix('fixed_k_')}"
    if method.startswith("dynamic_theta_"):
        return f"dynamic:{method.removeprefix('dynamic_theta_').replace('p', '.')}"
    head, _, delta = method.partition("noveto_delta_")
    variant = head.removeprefix("xadaptive").strip("_").replace("_", "-")
    value = f"{float(delta.replace('p', '.')):g}"
    return f"MI {variant} δ={value}" if variant else f"MI δ={value}"


def rate(cell: dict[str, dict], ids: set[str]) -> float:
    # File order, so on a whole cell this is the summary's number to the last bit.
    return corpus_rates([r for i, r in cell.items() if i in ids])["tokens_per_second"]


def score(cell: dict[str, dict], ids: set[str]) -> float:
    return sum(bool(cell[i]["score"]) for i in ids) / len(ids) if ids else float("nan")


def ratio(a: float, b: float) -> float:
    return a / b if b and b > 0 else float("nan")


def row_for(root: str, bench: str, method: str, cell: dict, base: dict[str, dict]) -> dict:
    common = set(cell).intersection(*(set(base[a]) for a in ANCHORS))
    own_capped = {i for i in common if capped(cell[i])}
    excluded = own_capped.union(*({i for i in common if capped(base[a][i])} for a in ANCHORS))
    paired = common - excluded
    f1 = base[FIXED1]
    f1_loop_free = {i for i in common if not capped(f1[i])}
    out = {
        "root": root, "benchmark": bench, "method": method, "label": label(method),
        "n": len(cell), "n_common": len(common),
        "n_capped": len(own_capped),
        "capped_correct": sum(bool(cell[i]["score"]) for i in own_capped),
        "n_correct": sum(bool(cell[i]["score"]) for i in common),
        "score": score(cell, common),
        "raw_tps": rate(cell, common), "raw_fixed1_tps": rate(f1, common),
        "loop_free_n": len(common) - len(own_capped), "loop_free_tps": rate(cell, common - own_capped),
        "loop_free_fixed1_n": len(f1_loop_free), "loop_free_fixed1_tps": rate(f1, f1_loop_free),
        "loop_free_score": score(cell, common - own_capped),
        "paired_n": len(paired), "paired_tps": rate(cell, paired),
        "paired_seconds_per_task": corpus_rates([r for i, r in cell.items() if i in paired])["seconds_per_task"],
        "paired_correct": sum(bool(cell[i]["score"]) for i in paired),
        "paired_score": score(cell, paired),
    }
    out["raw_speedup_vs_fixed1"] = ratio(out["raw_tps"], out["raw_fixed1_tps"])
    out["loop_free_speedup_vs_fixed1"] = ratio(out["loop_free_tps"], out["loop_free_fixed1_tps"])
    for anchor, tag in ((FIXED1, "fixed1"), (DYN95, "dyn95"), (DYN90, "dyn90")):
        out[f"paired_{tag}_tps"] = rate(base[anchor], paired)
        out[f"paired_{tag}_score"] = score(base[anchor], paired)
    out["paired_speedup_vs_fixed1"] = ratio(out["paired_tps"], out["paired_fixed1_tps"])
    out["paired_dyn95_speedup_vs_fixed1"] = ratio(out["paired_dyn95_tps"], out["paired_fixed1_tps"])
    out["paired_dyn90_speedup_vs_fixed1"] = ratio(out["paired_dyn90_tps"], out["paired_fixed1_tps"])
    return out


COLUMNS = [
    "root", "benchmark", "method", "label", "n", "n_common", "n_capped", "capped_correct", "n_correct", "score",
    "raw_tps", "raw_fixed1_tps", "raw_speedup_vs_fixed1",
    "loop_free_n", "loop_free_tps", "loop_free_fixed1_n", "loop_free_fixed1_tps",
    "loop_free_speedup_vs_fixed1", "loop_free_score",
    "paired_n", "paired_tps", "paired_seconds_per_task", "paired_correct", "paired_score",
    "paired_fixed1_tps", "paired_speedup_vs_fixed1", "paired_fixed1_score",
    "paired_dyn95_tps", "paired_dyn95_speedup_vs_fixed1", "paired_dyn95_score",
    "paired_dyn90_tps", "paired_dyn90_speedup_vs_fixed1", "paired_dyn90_score",
]


def check_against_summary(root: Path, rows: list[dict]) -> list[str]:
    """Where C is the whole cell, raw and loop-free tok/s must equal the root's summary.csv."""
    path = root / "comparison/summary.csv"
    if not path.exists():
        return [f"{path}: missing"]
    summary = {(r["benchmark"], r["method"]): r for r in csv.DictReader(path.open(newline="", encoding="utf-8"))}
    problems = []
    for row in rows:
        s = summary.get((row["benchmark"], row["method"]))
        if s is None or row["n_common"] != row["n"]:
            continue
        if float(s["tokens_per_second"]) != row["raw_tps"]:
            problems.append(f"{root.name} {row['benchmark']}/{row['method']}: raw tok/s differs from summary.csv")
        if "loop_free_tokens_per_second" in s and float(s["loop_free_tokens_per_second"]) != row["loop_free_tps"]:
            problems.append(f"{root.name} {row['benchmark']}/{row['method']}: loop-free tok/s differs from summary.csv")
    return problems


def tps_x(tps: float, speedup: float) -> str:
    return f"{tps:.1f} ({speedup:.2f}×)"


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def root_note(name: str, rows: list[dict]) -> str:
    sizes = {}
    for row in rows:
        sizes.setdefault(row["benchmark"], row["n"])
    shown = ", ".join(f"{BENCH_LABEL[b]} {n}" for b, n in sizes.items())
    return f"Tasks per cell: {shown}."


def markdown(groups: list[tuple[str, list[dict]]], capped_total: int, capped_right: int, problems: list[str]) -> str:
    lines = [
        "# Loop-free throughput: EC2 block-16 runs",
        "",
        f"Generated {datetime.date.today().isoformat()} by `MI_decoding/loop_free_report.py` from the "
        "records only (no GPU). SDAR-8B-b16 baselines and the MI rule (8B proposes, 1.7B verifies), "
        "all timed on EC2 A10G cards: compare them only with each other.",
        "",
        "Definitions (all rates are corpus tokens/s, prefill included; × is the ratio to fixed:1 "
        "on the same footing):",
        "",
        "```",
        "C          tasks of the cell that baseline fixed:1, dynamic:0.95 and dynamic:0.90 also ran",
        "raw        tok/s over C, runaways included (the summary's tokens_per_second for a whole cell)",
        "           x = raw / fixed:1 raw over the same C",
        "loop-free  tok/s over C minus the cell's own capped tasks",
        "           x = loop-free / fixed:1 over C minus fixed:1's own capped tasks",
        "P          C minus every task capped in {this method, fixed:1, dynamic:0.95, dynamic:0.90}",
        "paired     tok/s over P; fixed:1 and both dynamic arms re-aggregated on exactly P",
        "           x = paired / fixed:1 over P",
        "tok/s      sum(output_token_count) / sum(wall_time_seconds)",
        "```",
        "",
        "Columns: `capped (right)` is the cell's runaways in C and how many of them were graded correct; "
        "`score` is over C and `P score` over P (this arm, then fixed:1 on the same P), in percent. The last "
        "three columns are fixed:1 and the dynamic arms on the same P as the row.",
        "",
        f"Across every cell below, {capped_total} capped task runs, {capped_right} graded correct: "
        "leaving the runaways out removes only wrong answers, so the score over a cell's loop-free tasks "
        "is its correct count over fewer tasks (CSV `loop_free_score`).",
        "",
    ]
    if problems:
        lines += ["**Consistency check against comparison/summary.csv failed:**", ""]
        lines += [f"- {p}" for p in problems] + [""]
    else:
        lines += ["Consistency check: wherever C is the whole cell, raw and loop-free tok/s equal the "
                  "root's `comparison/summary.csv` (`tokens_per_second`, `loop_free_tokens_per_second`) exactly.", ""]
    for name, rows in groups:
        lines += [f"## {name}", "", root_note(name, rows), ""]
        for bench in BENCHMARKS:
            bench_rows = [r for r in rows if r["benchmark"] == bench]
            if not bench_rows:
                continue
            lines += [
                f"### {BENCH_LABEL[bench]}", "",
                "| method | n | capped (right) | score | raw tok/s (×) | loop-free tok/s (×) | P n "
                "| paired tok/s (×) | P score (this / fixed:1) | fixed:1 on P | dyn:0.95 on P (×) | dyn:0.90 on P (×) |",
                "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
            ]
            for r in bench_rows:
                lines.append(
                    f"| {r['label']} | {r['n_common']} | {r['n_capped']} ({r['capped_correct']}) | {pct(r['score'])} "
                    f"| {tps_x(r['raw_tps'], r['raw_speedup_vs_fixed1'])} "
                    f"| {tps_x(r['loop_free_tps'], r['loop_free_speedup_vs_fixed1'])} "
                    f"| {r['paired_n']} | {tps_x(r['paired_tps'], r['paired_speedup_vs_fixed1'])} "
                    f"| {pct(r['paired_score'])} / {pct(r['paired_fixed1_score'])} | {r['paired_fixed1_tps']:.1f} "
                    f"| {tps_x(r['paired_dyn95_tps'], r['paired_dyn95_speedup_vs_fixed1'])} "
                    f"| {tps_x(r['paired_dyn90_tps'], r['paired_dyn90_speedup_vs_fixed1'])} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    base = {bench: {m: records(BASELINE / bench / m) for m in BASE_METHODS} for bench in BENCHMARKS}
    groups: list[tuple[str, list[dict]]] = []
    problems: list[str] = []
    roots = [(BASELINE, BASE_METHODS)] + [(root, None) for root in MI_ROOTS]
    for root, methods in roots:
        rows = []
        for bench in BENCHMARKS:
            if not (root / bench).is_dir():
                continue
            names = methods or sorted(p.name for p in (root / bench).iterdir()
                                      if (p / "records.jsonl").exists())
            for method in names:
                rows.append(row_for(root.name, bench, method, records(root / bench / method), base[bench]))
        problems += check_against_summary(root, rows)
        title = f"{root.name} (baselines)" if root == BASELINE else root.name
        groups.append((title, rows))
    all_rows = [r for _, rows in groups for r in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "ec2_b16.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    capped_total = sum(r["n_capped"] for r in all_rows)
    capped_right = sum(r["capped_correct"] for r in all_rows)
    (OUT_DIR / "ec2_b16.md").write_text(markdown(groups, capped_total, capped_right, problems) + "\n",
                                        encoding="utf-8")
    print(f"Wrote {len(all_rows)} rows to {OUT_DIR / 'ec2_b16.csv'} and {OUT_DIR / 'ec2_b16.md'}; "
          f"{capped_total} capped task runs, {capped_right} correct; "
          f"{len(problems)} consistency problems")
    for p in problems:
        print("  ", p)


if __name__ == "__main__":
    main()
