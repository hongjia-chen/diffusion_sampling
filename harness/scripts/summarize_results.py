#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import math
import statistics
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

# Runnable from any directory: the package lives next to scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdar_harness.io import percentile, stream_jsonl, write_json

# Loop-free columns, appended after every older column in this order. A capped
# task (`maximum_length_termination`: no stop token before the canvas ran out)
# is usually a repetition loop: it costs minutes, it is the cheapest kind of
# token for a parallel decoder, and it scores wrong, so it inflates tokens/s
# and the speedup over fixed:1. These columns leave such tasks out.
#   per cell: the cell's own capped tasks are dropped;
#   paired:   each method against the same root and benchmark's fixed:1, on
#             the tasks both ran less any task capped in either, so the
#             speedup over fixed:1 compares the same tasks.
LOOP_FREE_FIELDS = (
    "n_capped",
    "loop_free_n",
    "loop_free_tokens_per_second",
    "loop_free_seconds_per_task",
    "loop_free_decode_tokens_per_second",
    "paired_loop_free_n",
    "paired_loop_free_tokens_per_second",
    "paired_loop_free_speedup_vs_fixed1",
)
FIXED1 = "fixed_k_1"


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def capped(row: dict[str, Any]) -> bool:
    return bool(row["maximum_length_termination"])


def corpus_rates(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """tokens/s, s/task and decode tokens/s over `rows`.

    The same formulas as summarize_file's all-task `tokens_per_second`,
    `seconds_per_task` and `decode_tokens_per_second` (corpus sum of tokens over
    sum of time; the mean task latency; prefill subtracted only when every task
    recorded it), so on the same rows they give the same numbers.
    """
    latencies = [float(row["wall_time_seconds"]) for row in rows]
    tokens = sum(float(row["output_token_count"]) for row in rows)
    has_prefill = all(row.get("prefill_seconds") is not None for row in rows)
    prefill = sum(float(row["prefill_seconds"]) for row in rows) if has_prefill else 0.0
    return {
        "tokens_per_second": tokens / sum(latencies) if sum(latencies) > 0 else float("nan"),
        "seconds_per_task": mean(latencies),
        "decode_tokens_per_second": (
            tokens / (sum(latencies) - prefill)
            if has_prefill and sum(latencies) > prefill else None
        ),
    }


def loop_free_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One cell's rates over its tasks that emitted a stop token."""
    kept = [row for row in rows if not capped(row)]
    rates = corpus_rates(kept)
    # Decode tokens/s only where the all-task column exists: every task, the
    # capped ones included, recorded its prefill.
    decode = all(row.get("prefill_seconds") is not None for row in rows)
    return {
        "n_capped": len(rows) - len(kept),
        "loop_free_n": len(kept),
        "loop_free_tokens_per_second": rates["tokens_per_second"],
        "loop_free_seconds_per_task": rates["seconds_per_task"],
        "loop_free_decode_tokens_per_second": rates["decode_tokens_per_second"] if decode else None,
    }


def paired_loop_free(cells: list[tuple[str, str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Paired loop-free columns for each (benchmark, method, records) cell.

    Each cell is paired with its benchmark's fixed:1 cell: P is the tasks both
    ran, less any task capped in either (for fixed:1 itself, its own uncapped
    tasks). The cell's tokens/s on P and its ratio to fixed:1's tokens/s on the
    same P. All three are None when the benchmark has no (or more than one)
    fixed:1 cell. Returned in the order of `cells`.
    """
    blank = {
        "paired_loop_free_n": None,
        "paired_loop_free_tokens_per_second": None,
        "paired_loop_free_speedup_vs_fixed1": None,
    }
    fixed1: dict[str, list[int]] = {}
    for index, (benchmark, method, _) in enumerate(cells):
        if method == FIXED1:
            fixed1.setdefault(benchmark, []).append(index)
    reference_rows = {
        benchmark: cells[indices[0]][2]
        for benchmark, indices in fixed1.items() if len(indices) == 1
    }
    out: list[dict[str, Any]] = []
    for benchmark, _, rows in cells:
        reference = reference_rows.get(benchmark)
        if reference is None:
            out.append(dict(blank))
            continue
        ids = {str(row["example_id"]) for row in rows}
        reference_ids = {str(row["example_id"]) for row in reference}
        excluded = {str(row["example_id"]) for row in rows + reference if capped(row)}
        keep = (ids & reference_ids) - excluded
        rate = corpus_rates([row for row in rows if str(row["example_id"]) in keep])
        reference_rate = corpus_rates(
            [row for row in reference if str(row["example_id"]) in keep]
        )["tokens_per_second"]
        out.append({
            "paired_loop_free_n": len(keep),
            "paired_loop_free_tokens_per_second": rate["tokens_per_second"],
            "paired_loop_free_speedup_vs_fixed1": (
                rate["tokens_per_second"] / reference_rate if reference_rate > 0 else float("nan")
            ),
        })
    return out


def stage_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Mean seconds per stage over the profiled tasks, plus each stage's share.

    Returns None when nothing in this method was profiled, so a summary from an
    ordinary run is unchanged rather than carrying a column of zeros that would
    read as "no time spent here".
    """
    profiled = [
        row for row in rows
        if row.get("stage_seconds") and row.get("profiled_wall_seconds")
    ]
    if not profiled:
        return None
    stages: dict[str, float] = {}
    calls: dict[str, int] = {}
    for row in profiled:
        for name, value in row["stage_seconds"].items():
            stages[name] = stages.get(name, 0.0) + float(value)
        for name, value in (row.get("stage_counts") or {}).items():
            calls[name] = calls.get(name, 0) + int(value)
    count = len(profiled)
    wall = sum(float(row["profiled_wall_seconds"]) for row in profiled) / count
    means = {name: total / count for name, total in stages.items()}
    accounted = sum(means.values())
    # Everything outside the timed stages: the Python loop and its bookkeeping.
    means["residual"] = wall - accounted
    out = {
        "tasks_profiled": count,
        "profiled_wall_seconds": wall,
        # The same tasks' timed (unprofiled) latency. Stage timers synchronize
        # at every boundary, so the profiled run is slower; the ratio says by
        # how much, and the shares are of the profiled run.
        "timed_wall_seconds": sum(float(row["wall_time_seconds"]) for row in profiled) / count,
        "seconds": means,
        "share": {
            name: (value / wall if wall > 0 else float("nan"))
            for name, value in means.items()
        },
    }
    out["profiling_slowdown"] = (
        wall / out["timed_wall_seconds"] if out["timed_wall_seconds"] > 0 else float("nan")
    )
    if calls:
        steps = sum(int(row["denoising_steps"]) for row in profiled)
        out["calls_per_task"] = {name: total / count for name, total in calls.items()}
        out["calls_per_step"] = {name: total / steps for name, total in calls.items()} if steps else {}
        out["ms_per_call"] = {
            name: 1000.0 * stages[name] / total
            for name, total in calls.items() if total and name in stages
        }
    return out


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
            "method_type", str(rows[0]["method"]).split("_")[0]
        ),
        "fixed_k": metadata.get("fixed_k"),
        "delta": metadata.get("delta"),
        "tau_nats": metadata.get("tau_nats"),
        "threshold": metadata.get("threshold"),
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
        "protocol": metadata.get("protocol", "legacy"),
        "number_of_tasks": len(rows),
        "number_scored": len(scores),
        "primary_metric": {
            "mbpp_plus": "MBPP+ pass@1, base tests (evalplus)",
            "humaneval_plus": "HumanEval pass@1, base tests (OpenCompass grading)",
            "humaneval": "HumanEval pass@1 (OpenCompass)",
            "mbpp_sanitized": "sanitized MBPP pass@1, 3 asserts (OpenCompass grading)",
            "gsm8k": "GSM8K accuracy",
            "math_500": "MATH-500 accuracy (math_verify)",
        }.get(str(rows[0]["benchmark"]), "Primary score"),
        "primary_score": mean([float(value) for value in scores]),
        "seconds_per_task": mean(latencies),
        "median_task_latency": statistics.median(latencies),
        "p95_task_latency": percentile(latencies, 0.95),
        # Corpus-level throughput: sum(tokens) / sum(time), NOT the mean of
        # per-task rates. vLLM and Fast-dLLM both aggregate this way, and it
        # matters -- averaging per-task ratios over-weights short tasks, which
        # on GSM8K means over-weighting exactly the collapsed early-EOS
        # completions whose speed is least meaningful.
        #
        # `tokens_per_second` is the industry-standard output-token throughput
        # (vLLM `total_output_tokens / elapsed`; Fast-dLLM `(ids != eos).sum()
        # / elapsed`) and the headline here (DESIGN.md F1). With early stop the
        # work done is the committed positions, not the canvas, so the
        # per-forward rate divides `decoded_positions`.
        "tokens_per_second": (
            sum(float(row["output_token_count"]) for row in rows) / sum(latencies)
            if sum(latencies) > 0 else float("nan")
        ),
        # Prompt prefill excluded: SimSD's "decoded tokens per second". Only
        # when every task recorded its prefill (runs before 2026-10-01 did not).
        "mean_prefill_seconds": (
            mean([float(row["prefill_seconds"]) for row in rows])
            if all(row.get("prefill_seconds") is not None for row in rows) else None
        ),
        "decode_tokens_per_second": (
            sum(float(row["output_token_count"]) for row in rows)
            / (sum(latencies) - sum(float(row["prefill_seconds"]) for row in rows))
            if all(row.get("prefill_seconds") is not None for row in rows)
            and sum(latencies) > sum(float(row["prefill_seconds"]) for row in rows)
            else None
        ),
        "decoded_positions_per_second": (
            sum(float(row.get("decoded_positions") or 0) for row in rows) / sum(latencies)
            if sum(latencies) > 0 else float("nan")
        ),
        "tokens_per_forward": (
            sum(float(row.get("decoded_positions") or 0) for row in rows)
            / sum(float(row["model_forward_passes"]) for row in rows)
            if sum(float(row.get("model_forward_passes") or 0) for row in rows) > 0
            else float("nan")
        ),
        "early_stop_rate": mean([float(bool(row.get("early_stopped"))) for row in rows]),
        "mean_model_forward_passes": mean(
            [float(row["model_forward_passes"]) for row in rows
             if row.get("model_forward_passes") is not None]
        ),
        "mean_denoising_steps": mean([float(row["denoising_steps"]) for row in rows]),
        # Latency decomposition, present only for tasks run with
        # --profile-stages. `residual` is the wall clock of the profiled
        # generation that none of the model stages account for: sampling,
        # masking and Python. Averaged over the profiled tasks only, so it is
        # NOT comparable to `seconds_per_task`, which is the median over all of
        # them from the unprofiled timed repeats.
        "stage_breakdown": stage_breakdown(rows),
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
        "max_candidates": max(int(row.get("max_candidates") or len(hist)) for row in rows),
        "mean_output_token_count": mean([float(row["output_token_count"]) for row in rows]),
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
        "plus_test_pass",
        "opencompass_pass",
        "answer_extraction_success",
    ):
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if values:
            summary[f"mean_{field}"] = mean(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--recorded-output-root",
        help="write records_path as if --output-root were this path: for a run "
        "copied from the host that made it (EC2), so a regenerated summary keeps "
        "the records_path it was first written with",
    )
    args = parser.parse_args()
    paths = []
    for records in args.output_root.rglob("records.jsonl"):
        scored = records.with_name("records.scored.jsonl")
        paths.append(scored if scored.exists() else records)
    if not paths:
        parser.error(f"No method records found below {args.output_root}")
    summaries = []
    cells = []
    for path in sorted(paths):
        summary = summarize_file(path)
        if args.recorded_output_root:
            relative = path.resolve().relative_to(args.output_root.resolve())
            summary["records_path"] = str(PurePosixPath(args.recorded_output_root) / relative)
        rows = list(stream_jsonl(path))
        summary.update(loop_free_summary(rows))
        summaries.append(summary)
        cells.append((summary["benchmark"], summary["method"], rows))
    for summary, paired in zip(summaries, paired_loop_free(cells)):
        summary.update(paired)
    summaries.sort(key=lambda row: (row["benchmark"], row["method"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summaries)
    # The loop-free columns go after every older column, so a summary made
    # before they existed is a prefix of one made now.
    all_fields: list[str] = []
    for row in summaries:
        for field in row:
            if field not in all_fields and field not in LOOP_FREE_FIELDS:
                all_fields.append(field)
    all_fields.extend(LOOP_FREE_FIELDS)
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
