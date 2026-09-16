#!/usr/bin/env python3
"""Exact trajectory diagnostics recoverable from completed benchmark records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "v3_matched"
REPORT = ROOT / "report"
DERIVED = REPORT / "derived"
FIGURES = REPORT / "figures"
BENCHMARKS = ("gsm8k", "mbpp_plus", "schema_json")
BENCHMARK_LABELS = {
    "gsm8k": "GSM8K",
    "mbpp_plus": "MBPP+",
    "schema_json": "Schema JSON",
}
DELTA_SLUGS = ("0p10", "0p15", "0p20", "0p30", "0p50", "0p75", "1p00", "1p50")
METHODS = tuple("adaptive_delta_" + value for value in DELTA_SLUGS)
WIDTH_BINS = (
    (1, 1, "1"),
    (2, 4, "2--4"),
    (5, 8, "5--8"),
    (9, 16, "9--16"),
    (17, 32, "17--32"),
    (33, 64, "33--64"),
)
ROW_END = r" \\"


def delta(method: str) -> str:
    return method.removeprefix("adaptive_delta_").replace("p", ".")


def records(benchmark: str, method: str) -> list[dict[str, Any]]:
    path = OUTPUT / benchmark / method / "records.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [str(row["example_id"]) for row in rows]
    if len(rows) != 100 or len(set(ids)) != 100:
        raise ValueError(f"expected 100 unique tasks in {path}, found {len(set(ids))}")
    return rows


def derive() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    survival: list[dict[str, Any]] = []
    widths: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        id_reference: set[str] | None = None
        for method in METHODS:
            rows = records(benchmark, method)
            ids = {str(row["example_id"]) for row in rows}
            if id_reference is None:
                id_reference = ids
            elif ids != id_reference:
                raise ValueError(f"adaptive methods are not paired for {benchmark}")
            steps = [int(row["denoising_steps"]) for row in rows]
            for step in range(1, max(steps) + 1):
                survival.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "delta": float(delta(method)),
                        "step": step,
                        "tasks_reached": sum(value >= step for value in steps),
                    }
                )
            histograms = [row["commitment_histogram"] for row in rows]
            width = max(len(values) for values in histograms)
            pooled = np.zeros(width, dtype=np.int64)
            for values in histograms:
                pooled[: len(values)] += np.asarray(values, dtype=np.int64)
            total_steps = int(pooled.sum())
            if total_steps != sum(steps):
                raise ValueError(f"histogram/step mismatch for {benchmark}/{method}")
            committed = int(sum((index + 1) * value for index, value in enumerate(pooled)))
            for lower, upper, label in WIDTH_BINS:
                count = int(pooled[lower - 1 : upper].sum())
                widths.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "delta": float(delta(method)),
                        "width_bin": label,
                        "step_count": count,
                        "step_share": count / total_steps,
                        "total_steps": total_steps,
                        "mean_commitment_width": committed / total_steps,
                        "at_cap_share": int(pooled[-1]) / total_steps,
                    }
                )
    return survival, widths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def survival_table(benchmark: str, rows: list[dict[str, Any]]) -> str:
    subset = [row for row in rows if row["benchmark"] == benchmark]
    maximum = max(int(row["step"]) for row in subset)
    lookup = {
        (str(row["method"]), int(row["step"])): int(row["tasks_reached"])
        for row in subset
    }
    header = "Step & " + " & ".join(f"$\\delta={delta(method)}$" for method in METHODS)
    body = []
    for step in range(1, maximum + 1):
        cells = [str(step)] + [
            str(lookup.get((method, step), 0)) for method in METHODS
        ]
        body.append(" & ".join(cells) + ROW_END)
    caption = (
        f"Adaptive-scheduler survival on {BENCHMARK_LABELS[benchmark]}. Each cell "
        "is the number of the 100 paired tasks whose generation reached that "
        "one-based denoising step."
    )
    return "\n".join(
        [
            r"\begingroup\scriptsize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\renewcommand{\arraystretch}{0.92}",
            r"\begin{longtable}{r*{8}{r}}",
            f"\\caption{{{caption}}}\\label{{tab:survival-{benchmark.replace('_', '-')}}}{ROW_END}",
            r"\toprule",
            header + ROW_END,
            r"\midrule",
            r"\endfirsthead",
            r"\multicolumn{9}{l}{\itshape Table \thetable\ (continued)}\\",
            r"\toprule",
            header + ROW_END,
            r"\midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endlastfoot",
            *body,
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
    )


def width_table(rows: list[dict[str, Any]]) -> str:
    lookup = {
        (str(row["benchmark"]), str(row["method"]), str(row["width_bin"])): row
        for row in rows
    }
    body = []
    for benchmark in BENCHMARKS:
        body.append(
            f"\\multicolumn{{9}}{{l}}{{\\textbf{{{BENCHMARK_LABELS[benchmark]}}}}}{ROW_END}"
        )
        for method in METHODS:
            sample = lookup[(benchmark, method, "1")]
            shares = [
                float(lookup[(benchmark, method, label)]["step_share"])
                for _, _, label in WIDTH_BINS
            ]
            cells = [f"$\\delta={delta(method)}$"] + [f"{100 * value:.1f}" for value in shares]
            cells += [
                f"{float(sample['mean_commitment_width']):.2f}",
                f"{100 * float(sample['at_cap_share']):.1f}",
            ]
            body.append(" & ".join(cells) + ROW_END)
        if benchmark != BENCHMARKS[-1]:
            body.append(r"\addlinespace[3pt]")
    header = (
        r"Method & 1 & 2--4 & 5--8 & 9--16 & 17--32 & 33--64 "
        r"& Mean & At cap"
    )
    caption = (
        "Distribution of adaptive commitment widths, pooled over all denoising "
        "steps from the 100 tasks. Width-bin and cap entries are percentages of "
        "steps; Mean is tokens committed per step. The 64-token cap rate diagnoses "
        "right-censoring by the configured proposal limit."
    )
    return "\n".join(
        [
            r"\begin{table}[!ht]",
            r"\centering\small",
            f"\\caption{{{caption}}}\\label{{tab:commitment-widths}}",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            header + ROW_END,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def plot(survival: list[dict[str, Any]], widths: list[dict[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
        }
    )
    colors = plt.cm.viridis(np.linspace(0.04, 0.94, len(METHODS)))
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.2), constrained_layout=True)
    for row_index, benchmark in enumerate(BENCHMARKS):
        ax = axes[row_index, 0]
        for method, color in zip(METHODS, colors):
            values = [
                row for row in survival
                if row["benchmark"] == benchmark and row["method"] == method
            ]
            ax.step(
                [row["step"] for row in values],
                [row["tasks_reached"] for row in values],
                where="post",
                color=color,
                linewidth=1.35,
                label=f"$\\delta={delta(method)}$",
            )
        ax.set_ylim(0, 103)
        ax.set_xlabel("denoising step")
        ax.set_ylabel("tasks reaching step (of 100)")
        ax.set_title(f"{BENCHMARK_LABELS[benchmark]}: trajectory survival")
        if row_index == 0:
            ax.legend(ncol=4, fontsize=7, loc="upper right")

        heat = np.zeros((len(METHODS), len(WIDTH_BINS)))
        for i, method in enumerate(METHODS):
            for j, (_, _, label) in enumerate(WIDTH_BINS):
                match = next(
                    row for row in widths
                    if row["benchmark"] == benchmark
                    and row["method"] == method
                    and row["width_bin"] == label
                )
                heat[i, j] = 100 * float(match["step_share"])
        ax = axes[row_index, 1]
        image = ax.imshow(heat, aspect="auto", cmap="Blues", vmin=0, vmax=60)
        ax.grid(False)
        ax.set_xticks(range(len(WIDTH_BINS)))
        ax.set_xticklabels([label for _, _, label in WIDTH_BINS])
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([f"$\\delta={delta(method)}$" for method in METHODS])
        ax.set_xlabel("tokens committed in a step")
        ax.set_title(f"{BENCHMARK_LABELS[benchmark]}: share of all steps")
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                color = "white" if heat[i, j] >= 32 else "black"
                ax.text(j, i, f"{heat[i, j]:.0f}", ha="center", va="center", color=color, fontsize=7)
    colorbar = fig.colorbar(image, ax=axes[:, 1], shrink=0.68, pad=0.02)
    colorbar.set_label("percent of denoising steps")
    fig.suptitle(
        "Adaptive trajectories from saved step counts and commitment histograms",
        fontsize=11,
        fontweight="bold",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig6_adaptive_trajectories.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    survival, widths = derive()
    write_csv(DERIVED / "adaptive_survival.csv", survival)
    write_csv(DERIVED / "adaptive_commitment_widths.csv", widths)
    for benchmark in BENCHMARKS:
        (REPORT / f"table_survival_{benchmark}.tex").write_text(
            survival_table(benchmark, survival), encoding="utf-8"
        )
    (REPORT / "table_commitment_widths.tex").write_text(width_table(widths), encoding="utf-8")
    plot(survival, widths)
    print("wrote adaptive survival/width CSVs, four tables, and fig6_adaptive_trajectories")


if __name__ == "__main__":
    main()
