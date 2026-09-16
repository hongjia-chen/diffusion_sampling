#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from dream_ablation.io import percentile, stream_jsonl, write_json


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def summarize_file(path: Path) -> dict[str, Any]:
    rows = list(stream_jsonl(path))
    if not rows:
        raise ValueError(f"No records in {path}")
    metadata_path = path.with_name("metadata.json")
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scores = [bool(row["score"]) for row in rows if row.get("score") is not None]
    latencies = [float(row["wall_time_seconds"]) for row in rows]
    # Sized from the data, not hardcoded: max_candidates is a run parameter and
    # the old key name baked in a cap of 8.
    width = max(
        len(row.get("commitment_histogram") or row["commitment_histogram_1_to_8"])
        for row in rows
    )
    hist = [0] * width
    failures: Counter[str] = Counter()
    for row in rows:
        values = row.get("commitment_histogram") or row["commitment_histogram_1_to_8"]
        for index, value in enumerate(values):
            hist[index] += int(value)
        if row.get("failure_type"):
            failures[str(row["failure_type"])] += 1
    accepted_total = sum((index + 1) * count for index, count in enumerate(hist))
    decision_total = sum(hist)
    summary = {
        "benchmark": rows[0]["benchmark"],
        "method": rows[0]["method"],
        "method_type": metadata.get(
            "method_type", "adaptive" if rows[0]["method"].startswith("adaptive") else "fixed"
        ),
        "fixed_k": metadata.get("fixed_k"),
        "delta": metadata.get("delta"),
        "tau_nats": metadata.get("tau_nats"),
        "temperature": metadata.get("temperature"),
        "top_p": metadata.get("top_p"),
        "confidence_metric": metadata.get("confidence_metric"),
        "model": metadata.get("model"),
        "model_revision": metadata.get("model_revision"),
        "dtype": metadata.get("dtype"),
        "proposer_is_verifier": metadata.get("proposer_is_verifier"),
        "continuation_length": metadata.get("continuation_length"),
        "max_candidates": metadata.get("max_candidates"),
        "verifier_microbatch_size": metadata.get("verifier_microbatch_size"),
        "timing_repeats": metadata.get("timing_repeats"),
        "prompt_template": metadata.get("prompt_template"),
        "number_of_tasks": len(rows),
        "number_scored": len(scores),
        "primary_metric": {
            "mbpp_plus": "MBPP+ pass@1",
            "gsm8k": "GSM8K exact match",
            "schema_json": "JSON exact match",
        }.get(str(rows[0]["benchmark"]), "Primary score"),
        "primary_score": mean([float(value) for value in scores]),
        "seconds_per_task": mean(latencies),
        "median_task_latency": statistics.median(latencies),
        "p95_task_latency": percentile(latencies, 0.95),
        "mean_denoising_steps": mean([float(row["denoising_steps"]) for row in rows]),
        "mean_dream_proposal_passes": mean([float(row["dream_proposal_passes"]) for row in rows]),
        # Batching-invariant verifier cost. Prefer this and the latencies above
        # when comparing schedulers.
        "mean_verifier_rows": mean([float(row["verifier_rows"]) for row in rows]),
        "mean_committed_tokens_per_step": (
            accepted_total / decision_total if decision_total else float("nan")
        ),
        "commitment_histogram": hist,
        # Share of steps that committed the maximum allowed. A non-trivial value
        # means the cap, not the delta criterion, is deciding those steps, and
        # the tokens-per-step axis is censored there.
        "at_cap_rate": (hist[-1] / decision_total) if decision_total else float("nan"),
        "max_candidates": max(int(row.get("max_candidates", len(hist))) for row in rows),
        "mean_output_token_count": mean([float(row["output_token_count"]) for row in rows]),
        "early_eos_rate": mean(
            [float(bool(row.get("early_eos", row.get("premature_eos")))) for row in rows]
        ),
        "maximum_length_rate": mean(
            [float(bool(row["maximum_length_termination"])) for row in rows]
        ),
        # Both of these move with --verifier-microbatch-size for identical work,
        # so they describe the batching configuration rather than the scheduler.
        # Diagnostics only; never compare schedulers on them.
        "batching_dependent": {
            "mean_verifier_forward_passes": mean(
                [float(row["verifier_forward_passes"]) for row in rows]
            ),
            "peak_gpu_memory_bytes": max(
                int(row["peak_gpu_memory_bytes"]) for row in rows
            ),
            "verifier_microbatch_size": metadata.get("verifier_microbatch_size"),
        },
        "failure_counts": dict(failures),
        "records_path": str(path.resolve()),
    }
    for field in (
        "syntax_valid",
        "base_test_pass",
        "mbpp_plus_pass",
        "answer_extraction_success",
        "json_parse_valid",
        "schema_valid",
        "field_accuracy",
    ):
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if values:
            summary[f"mean_{field}"] = mean(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for records in args.output_root.rglob("records.jsonl"):
        scored = records.with_name("records.scored.jsonl")
        paths.append(scored if scored.exists() else records)
    if not paths:
        parser.error(f"No method records found below {args.output_root}")
    summaries = [summarize_file(path) for path in sorted(paths)]
    summaries.sort(key=lambda row: (row["benchmark"], row["method"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summaries)
    all_fields: list[str] = []
    for row in summaries:
        for field in row:
            if field not in all_fields:
                all_fields.append(field)
    flat_rows = []
    for row in summaries:
        flat = dict(row)
        flat["commitment_histogram"] = json.dumps(flat["commitment_histogram"])
        flat["failure_counts"] = json.dumps(flat["failure_counts"], sort_keys=True)
        flat_rows.append(flat)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote {len(summaries)} method summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
