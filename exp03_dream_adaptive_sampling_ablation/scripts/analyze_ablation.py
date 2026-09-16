#!/usr/bin/env python3
"""Aggregate the completed scheduler sweep and generate report-ready plots/tables."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "core"
REPORT_ROOT = ROOT / "report"
FIGURE_ROOT = REPORT_ROOT / "figures"
BENCHMARKS = ("mbpp_plus", "gsm8k", "schema_json")
FIXED = ("fixed_k_1", "fixed_k_2", "fixed_k_4", "fixed_k_8")

METRIC_LABELS = {
    "score": "Primary score",
    "base_test_pass": "MBPP base-test pass rate",
    "mbpp_plus_pass": "MBPP+ pass rate",
    "syntax_valid": "Python syntax-valid rate",
    "answer_extraction_success": "Answer-extraction success rate",
    "json_parse_valid": "JSON parse-valid rate",
    "schema_valid": "Schema-valid rate",
    "field_accuracy": "Field accuracy",
    "output_token_count": "Output token count",
    "eos_position": "EOS position",
    "denoising_steps": "Denoising steps",
    "dream_proposal_passes": "Dream proposal passes",
    "verifier_rows": "Verifier rows",
    "verifier_forward_passes": "Verifier forward passes",
    "peak_gpu_memory_bytes": "Peak GPU memory (GiB)",
    "mean_committed_tokens_per_step": "Committed tokens per step",
    "maximum_length_termination": "Maximum-length termination rate",
    "premature_eos": "Premature-EOS rate",
    **{f"commitment_share_{k}": f"Share of steps committing {k} token{'s' if k != 1 else ''}"
       for k in range(1, 9)},
}

RATE_METRICS = {
    "score", "base_test_pass", "mbpp_plus_pass", "syntax_valid",
    "answer_extraction_success", "json_parse_valid", "schema_valid",
    "field_accuracy", "maximum_length_termination", "premature_eos",
    *(f"commitment_share_{k}" for k in range(1, 9)),
}

# Logged measurements/configuration are intentionally distinguished. These are
# the per-example measurements for which a cross-scheduler average is meaningful.
COMMON_METRICS = (
    "score", "output_token_count", "eos_position", "denoising_steps",
    "dream_proposal_passes", "verifier_rows", "verifier_forward_passes",
    "peak_gpu_memory_bytes", "mean_committed_tokens_per_step",
    "maximum_length_termination", "premature_eos",
    *(f"commitment_share_{k}" for k in range(1, 9)),
)
BENCHMARK_METRICS = {
    "mbpp_plus": ("base_test_pass", "mbpp_plus_pass", "syntax_valid"),
    "gsm8k": ("answer_extraction_success",),
    "schema_json": ("json_parse_valid", "schema_valid", "field_accuracy"),
}
QUALITY_METRICS = {
    "mbpp_plus": ("mbpp_plus_pass", "base_test_pass", "syntax_valid"),
    "gsm8k": ("score", "answer_extraction_success"),
    "schema_json": ("score", "json_parse_valid", "schema_valid", "field_accuracy"),
}


def method_sort_key(method: str) -> tuple[int, float]:
    if method.startswith("fixed_k_"):
        return (0, float(method.rsplit("_", 1)[1]))
    match = re.fullmatch(r"adaptive_delta_(\d+)p(\d+)", method)
    if not match:
        raise ValueError(f"Unrecognized scheduler directory: {method}")
    return (1, float(f"{match.group(1)}.{match.group(2)}"))


def method_label(method: str) -> str:
    if method.startswith("fixed_k_"):
        return f"fixed k={method.rsplit('_', 1)[1]}"
    match = re.fullmatch(r"adaptive_delta_(\d+)p(\d+)", method)
    assert match
    return f"adaptive δ={match.group(1)}.{match.group(2)}"


def method_style(method: str) -> tuple[str, str]:
    order, value = method_sort_key(method)
    if order == 0:
        colors = {1.0: "#1b9e77", 2.0: "#66a61e", 4.0: "#e6ab02", 8.0: "#d95f02"}
        return colors[value], "s"
    deltas = (0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50)
    return plt.get_cmap("viridis")(0.08 + 0.84 * deltas.index(value) / (len(deltas) - 1)), "o"


def load_records(benchmark: str, method_dir: Path) -> list[dict[str, Any]]:
    filename = "records.scored.jsonl" if benchmark == "mbpp_plus" else "records.jsonl"
    path = method_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float, bool))]


def mean_ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def aggregate() -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods_by_benchmark: dict[str, list[str]] = {}
    validation: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        method_dirs = sorted(
            (path for path in (OUTPUT_ROOT / benchmark).iterdir() if path.is_dir()),
            key=lambda path: method_sort_key(path.name),
        )
        methods = [path.name for path in method_dirs]
        methods_by_benchmark[benchmark] = methods
        id_sets: dict[str, set[str]] = {}
        validation[benchmark] = {}
        for method_dir in method_dirs:
            records = load_records(benchmark, method_dir)
            for record in records:
                histogram = record.get("commitment_histogram") or record.get("commitment_histogram_1_to_8", [])
                total = sum(histogram)
                for k in range(1, 9):
                    record[f"commitment_share_{k}"] = histogram[k - 1] / total if total else 0.0
            method = method_dir.name
            ids = [str(record["example_id"]) for record in records]
            id_sets[method] = set(ids)
            validation[benchmark][method] = {
                "records": len(records),
                "unique_example_ids": len(set(ids)),
                "duplicate_records": len(ids) - len(set(ids)),
            }
            if len(records) != 100 or len(set(ids)) != 100:
                raise ValueError(f"{benchmark}/{method}: expected 100 unique records, got {len(records)}/{len(set(ids))}")
            wall, wall_ci = mean_ci(numeric([record.get("wall_time_seconds") for record in records]))
            for metric in COMMON_METRICS + BENCHMARK_METRICS[benchmark]:
                values = numeric([record.get(metric) for record in records])
                if not values:
                    continue
                if metric == "peak_gpu_memory_bytes":
                    values = [value / 1024**3 for value in values]
                value, value_ci = mean_ci(values)
                rows.append({
                    "benchmark": benchmark,
                    "method": method,
                    "method_label": method_label(method),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "mean_metric": value,
                    "ci95_metric": value_ci,
                    "mean_wall_time_seconds": wall,
                    "ci95_wall_time_seconds": wall_ci,
                    "number_of_tasks": len(records),
                })
        reference = id_sets[methods[0]]
        paired = all(ids == reference for ids in id_sets.values())
        validation[benchmark]["paired_example_sets"] = paired
        if not paired:
            raise ValueError(f"{benchmark}: scheduler example sets differ")
    return rows, methods_by_benchmark, validation


def slug(metric: str) -> str:
    return metric.replace("_", "-")


def plot_metric(benchmark: str, metric: str, subset: list[dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 6.0), constrained_layout=True)
    for row in sorted(subset, key=lambda item: method_sort_key(item["method"])):
        color, marker = method_style(row["method"])
        scale = 100.0 if metric in RATE_METRICS else 1.0
        ax.errorbar(
            row["mean_metric"] * scale,
            row["mean_wall_time_seconds"],
            xerr=row["ci95_metric"] * scale,
            yerr=row["ci95_wall_time_seconds"],
            fmt=marker, color=color, ecolor=color, markersize=7,
            elinewidth=1, capsize=2, alpha=0.95,
        )
        ax.annotate(row["method_label"], (row["mean_metric"] * scale, row["mean_wall_time_seconds"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7)
    xlabel = METRIC_LABELS[metric] + (" (%)" if metric in RATE_METRICS else "")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean wall-clock time per sample (s)")
    ax.set_title(f"{benchmark.replace('_', ' ').upper()}: {METRIC_LABELS[metric]} vs. latency")
    ax.grid(True, alpha=0.25)
    out = FIGURE_ROOT / benchmark / f"{slug(metric)}-vs-wall-time.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return out


def plot_quality_overview(benchmark: str, rows: list[dict[str, Any]]) -> Path:
    metrics = QUALITY_METRICS[benchmark]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 5.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        subset = [row for row in rows if row["benchmark"] == benchmark and row["metric"] == metric]
        for row in sorted(subset, key=lambda item: method_sort_key(item["method"])):
            color, marker = method_style(row["method"])
            ax.errorbar(row["mean_metric"] * 100, row["mean_wall_time_seconds"],
                        xerr=row["ci95_metric"] * 100, yerr=row["ci95_wall_time_seconds"],
                        fmt=marker, color=color, markersize=6, capsize=2, elinewidth=0.9)
        ax.set_xlabel(METRIC_LABELS[metric] + " (%)")
        ax.set_ylabel("Mean wall-clock time/sample (s)")
        ax.grid(True, alpha=0.25)
    handles = []
    methods = sorted({row["method"] for row in rows if row["benchmark"] == benchmark}, key=method_sort_key)
    for method in methods:
        color, marker = method_style(method)
        handles.append(Line2D([], [], marker=marker, linestyle="none", color=color,
                              label=method_label(method), markersize=6))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{benchmark.replace('_', ' ').upper()}: quality–latency trade-offs", fontsize=13)
    out = FIGURE_ROOT / f"{benchmark}-quality-overview.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return out


def write_csv(rows: list[dict[str, Any]]) -> None:
    path = REPORT_ROOT / "aggregated_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("δ", r"$\delta$")


def write_primary_tables(rows: list[dict[str, Any]]) -> None:
    lines = []
    for benchmark in BENCHMARKS:
        metric = "mbpp_plus_pass" if benchmark == "mbpp_plus" else "score"
        subset = sorted((row for row in rows if row["benchmark"] == benchmark and row["metric"] == metric),
                        key=lambda row: method_sort_key(row["method"]))
        lines.extend([
            r"\begin{table}[htbp]", r"\centering", r"\small",
            rf"\caption{{Primary performance and mean latency on {tex_escape(benchmark)} ($n=100$ paired examples per scheduler).}}",
            rf"\label{{tab:{benchmark}-primary}}",
            r"\begin{tabular}{lrr}", r"\toprule",
            "Scheduler & Performance (\\%) & Time/sample (s) " + r"\\", r"\midrule",
        ])
        for row in subset:
            lines.append(f"{tex_escape(row['method_label'])} & {100*row['mean_metric']:.1f} & {row['mean_wall_time_seconds']:.2f} " + r"\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (REPORT_ROOT / "primary_tables.tex").write_text("\n".join(lines))


def write_manifest(rows: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    metrics = {benchmark: sorted({row["metric"] for row in rows if row["benchmark"] == benchmark})
               for benchmark in BENCHMARKS}
    (REPORT_ROOT / "analysis_manifest.json").write_text(json.dumps({
        "output_root": str(OUTPUT_ROOT),
        "metrics": metrics,
        "validation": validation,
        "notes": {
            "aggregation": "arithmetic mean across 100 paired examples",
            "error_bars": "normal-approximation 95% confidence interval, 1.96 * sample standard error",
            "wall_clock_axis": "mean of per-record wall_time_seconds",
            "mbpp_source": "records.scored.jsonl (EvalPlus results)",
        },
    }, indent=2) + "\n")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "pdf.fonttype": 42})
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, _methods, validation = aggregate()
    write_csv(rows)
    write_primary_tables(rows)
    write_manifest(rows, validation)
    for benchmark in BENCHMARKS:
        plot_quality_overview(benchmark, rows)
        for metric in sorted({row["metric"] for row in rows if row["benchmark"] == benchmark}):
            plot_metric(benchmark, metric,
                        [row for row in rows if row["benchmark"] == benchmark and row["metric"] == metric])
    print(f"Wrote {len(rows)} scheduler-metric aggregates to {REPORT_ROOT}")


if __name__ == "__main__":
    main()
