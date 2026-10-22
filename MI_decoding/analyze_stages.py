"""Per-stage latency of each arm, from the stage-profiled tasks (PROFILE_STAGES).

For every benchmark and arm under an output root: the profiled tasks' seconds
per task in each stage and its share, milliseconds per call and calls per
step, the untimed residual (the Python loop), and how much slower the
profiled decode ran than the timed one. Model load and warmup times come from
the run metadata. Writes `<output-root>/comparison/stage_breakdown.csv`.

    python analyze_stages.py --output-root outputs/mi/8b_proposes
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT.parent / "harness" / "scripts"))
from summarize_results import stage_breakdown  # noqa: E402

ORDER = ("prefill", "propose_forward", "propose_rank", "verify_forward", "verify_test",
         "verifier", "decide", "stop_check", "commit", "remask", "residual")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=PROJECT / "outputs/mi/8b_proposes")
    args = parser.parse_args()

    table: list[dict] = []
    for bench_dir in sorted(p for p in args.output_root.iterdir() if p.is_dir() and p.name != "comparison"):
        loads = [json.loads(p.read_text()).get("load_seconds") for p in bench_dir.glob("run_metadata_worker_*.json")]
        loads = [value for value in loads if value]
        for method_dir in sorted(p for p in bench_dir.iterdir() if (p / "records.jsonl").exists()):
            rows = [json.loads(line) for line in (method_dir / "records.jsonl").open()]
            breakdown = stage_breakdown(rows)
            if breakdown is None:
                continue
            meta = json.loads((method_dir / "metadata.json").read_text())
            warmups = meta.get("warmup_seconds") or []
            row = {
                "benchmark": bench_dir.name, "arm": method_dir.name,
                "tasks_profiled": breakdown["tasks_profiled"],
                "timed_wall_s": breakdown["timed_wall_seconds"],
                "profiled_wall_s": breakdown["profiled_wall_seconds"],
                "profiling_slowdown": breakdown["profiling_slowdown"],
                "warmup_s_each": sum(warmups) / len(warmups) if warmups else None,
                "load_s_proposer": loads[-1].get("proposer") if loads else None,
                "load_s_verifier": loads[-1].get("verifier") if loads else None,
            }
            for stage in ORDER:
                if stage in breakdown["seconds"]:
                    row[f"s_{stage}"] = breakdown["seconds"][stage]
                    row[f"share_{stage}"] = breakdown["share"][stage]
                for key, prefix in (("ms_per_call", "ms_"), ("calls_per_step", "per_step_")):
                    if stage in breakdown.get(key, {}):
                        row[f"{prefix}{stage}"] = breakdown[key][stage]
            table.append(row)
    if not table:
        raise SystemExit(f"no stage-profiled records under {args.output_root}")

    out_dir = args.output_root / "comparison"
    out_dir.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(key for row in table for key in row))
    with (out_dir / "stage_breakdown.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    stages = [stage for stage in ORDER if any(f"s_{stage}" in row for row in table)]
    short = {"prefill": "prefill", "propose_forward": "prop_fwd", "propose_rank": "prop_rank",
             "verify_forward": "ver_fwd", "verify_test": "ver_test", "verifier": "verifier",
             "decide": "decide", "stop_check": "stop", "commit": "commit", "remask": "remask",
             "residual": "residual"}
    for bench in dict.fromkeys(row["benchmark"] for row in table):
        rows = [row for row in table if row["benchmark"] == bench]
        print(f"\n== {bench}: share of the profiled decode (seconds per task in brackets)")
        print(f"{'arm':30s} {'n':>3s} {'timed s':>8s} {'slowdn':>6s} " + " ".join(f"{short[s]:>15s}" for s in stages))
        for row in rows:
            cells = []
            for stage in stages:
                if f"s_{stage}" in row:
                    cells.append(f"{100 * row[f'share_{stage}']:5.1f}% ({row[f's_{stage}']:5.2f})")
                else:
                    cells.append("")
            print(f"{row['arm']:30s} {row['tasks_profiled']:3d} {row['timed_wall_s']:8.2f} "
                  f"{row['profiling_slowdown']:6.2f} " + " ".join(f"{c:>15s}" for c in cells))
        print(f"-- {bench}: milliseconds per call (calls per step)")
        for row in rows:
            cells = []
            for stage in stages:
                if f"ms_{stage}" in row:
                    cells.append(f"{row[f'ms_{stage}']:6.1f} ({row.get(f'per_step_{stage}', 0):4.2f})")
                else:
                    cells.append("")
            print(f"{row['arm']:30s} {'':3s} {'':8s} {'':6s} " + " ".join(f"{c:>15s}" for c in cells))
        first = rows[0]
        print(f"-- one-off: load {first['load_s_proposer'] or 0:.1f} s proposer, "
              f"{first['load_s_verifier'] or 0:.1f} s verifier; warmup per run "
              + ", ".join(f"{row['arm']} {row['warmup_s_each']:.1f} s" for row in rows if row['warmup_s_each']))
    print(f"\nwrote {out_dir / 'stage_breakdown.csv'}")


if __name__ == "__main__":
    main()
