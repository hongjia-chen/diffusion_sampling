"""Tokens per step against δ for the MI rule, with the b64 baselines on the same tasks.

Reads a launch_mi.sh output root (one directory per benchmark, one per δ arm)
and, for the same example ids, the SDAR baselines of the proposer's
checkpoint. Writes `<probe-root>/comparison/delta_probe.csv` and prints one
table per benchmark.

Columns:
  committed/step  positions committed per proposer step (the histogram; counts
                  post-stop positions inside the last block)
  out/step        output tokens (up to the stop token) per step
  lone            share of steps with a single masked slot left (forced 1)
  block-bound     share of steps where the rule passed every remaining slot of
                  several, so only the block stopped it
  vpass/step      verifier forwards per step

    python analyze_delta_probe.py
    python analyze_delta_probe.py --probe-root outputs/mi_probe/8b_proposes
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
BASELINE_ARMS = ("fixed_k_1", "dynamic_theta_0p95", "dynamic_theta_0p90", "fixed_k_2", "fixed_k_4", "fixed_k_8")


def load_records(method_dir: Path) -> dict[str, dict]:
    scored = method_dir / "records.scored.jsonl"
    path = scored if scored.exists() else method_dir / "records.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.open()]
    return {str(row["example_id"]): row for row in rows}


def correct(row: dict) -> bool | None:
    value = row.get("opencompass_pass", row.get("score"))
    if value is None or value == "None":
        return None
    return value in (True, 1, 1.0, "True")


def summarize(rows: list[dict]) -> dict:
    steps = sum(row["denoising_steps"] for row in rows)
    committed = sum((i + 1) * c for row in rows for i, c in enumerate(row["commitment_histogram"]))
    verdicts = [correct(row) for row in rows]
    graded = [v for v in verdicts if v is not None]
    out = {
        "n": len(rows),
        "accuracy": sum(graded) / len(graded) if graded else math.nan,
        "graded": len(graded),
        "committed_per_step": committed / steps if steps else math.nan,
        "output_per_step": sum(row["output_token_count"] for row in rows) / steps if steps else math.nan,
        "steps_per_task": steps / len(rows),
        "output_tokens": sum(row["output_token_count"] for row in rows) / len(rows),
        "verifier_passes_per_step": sum(row.get("verifier_forward_passes", 0) for row in rows) / steps,
        "tokens_per_second": sum(row["output_token_count"] for row in rows)
        / sum(row["wall_time_seconds"] for row in rows),
        "max_length_rate": sum(bool(row["maximum_length_termination"]) for row in rows) / len(rows),
    }
    if all("lone_candidate_steps" in row for row in rows):
        out["lone_share"] = sum(row["lone_candidate_steps"] for row in rows) / steps
        out["block_bound_share"] = sum(row["all_candidates_accepted_steps"] for row in rows) / steps
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-root", type=Path, default=PROJECT / "outputs/mi_probe/8b_proposes")
    parser.add_argument("--baseline-root", type=Path, default=PROJECT / "outputs/opencompass/SDAR-8B-Chat-b64")
    args = parser.parse_args()

    table: list[dict] = []
    for bench_dir in sorted(p for p in args.probe_root.iterdir() if p.is_dir() and p.name != "comparison"):
        arms = []
        for method_dir in bench_dir.iterdir():
            if not method_dir.name.startswith("xadaptive") or not (method_dir / "metadata.json").exists():
                continue
            records = load_records(method_dir)
            if not records:
                continue
            meta = json.loads((method_dir / "metadata.json").read_text())
            arms.append((float(meta["delta"]), method_dir.name, records))
        if not arms:
            continue
        arms.sort()
        # An arm that stopped early (fewer tasks than the fullest) is left out;
        # the rest are compared on the tasks they all finished.
        most = max(len(records) for _, _, records in arms)
        for _, name, records in arms:
            if len(records) < most:
                print(f"skipping {bench_dir.name}/{name}: {len(records)} of {most} tasks")
        arms = [arm for arm in arms if len(arm[2]) == most]
        ids = set.intersection(*(set(records) for _, _, records in arms))
        for delta, name, records in arms:
            row = {"benchmark": bench_dir.name, "arm": name, "delta": delta,
                   "tau_nats": math.log1p(delta), **summarize([records[i] for i in sorted(ids)])}
            table.append(row)
        for arm in BASELINE_ARMS:
            records = load_records(args.baseline_root / bench_dir.name / arm)
            if ids and ids <= set(records):
                table.append({"benchmark": bench_dir.name, "arm": arm, "delta": math.nan,
                              "tau_nats": math.nan, **summarize([records[i] for i in sorted(ids)])})

    if not table:
        raise SystemExit(f"no xadaptive records under {args.probe_root}")
    out_dir = args.probe_root / "comparison"
    out_dir.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(key for row in table for key in row))
    with (out_dir / "delta_probe.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    def fmt(value, spec):
        return "" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)

    for bench in dict.fromkeys(row["benchmark"] for row in table):
        rows = [row for row in table if row["benchmark"] == bench]
        print(f"\n== {bench} ({rows[0]['n']} tasks)")
        print(f"{'arm':30s} {'tau':>6s} {'acc':>5s} {'commit/st':>9s} {'out/st':>6s} {'steps':>6s} "
              f"{'lone':>5s} {'blkbd':>5s} {'vp/st':>5s} {'tok/s':>6s} {'maxlen':>6s}")
        for row in rows:
            print(f"{row['arm']:30s} {fmt(row['tau_nats'], '6.3f'):>6s} {fmt(row['accuracy'], '5.2f'):>5s} "
                  f"{row['committed_per_step']:9.2f} {row['output_per_step']:6.2f} {row['steps_per_task']:6.1f} "
                  f"{fmt(row.get('lone_share'), '5.2f'):>5s} {fmt(row.get('block_bound_share'), '5.2f'):>5s} "
                  f"{row['verifier_passes_per_step']:5.2f} {row['tokens_per_second']:6.1f} {row['max_length_rate']:6.2f}")
    print(f"\nwrote {out_dir / 'delta_probe.csv'}")


if __name__ == "__main__":
    main()
