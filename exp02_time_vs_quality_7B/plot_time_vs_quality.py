#!/usr/bin/env python3
"""Plot fixed-k and adaptive Dream schedules on one time-versus-quality figure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--adaptive-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--log-time-axis", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error("--dpi must be positive.")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No summary rows found in {path}.")
    return rows


def normalize_fixed(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "schedule_type": "fixed",
            "label": f"k={int(row['k'])}",
            "schedule_value": int(row["k"]),
            "quality": float(row["MAUVE"]),
            "time": float(row["mean_amortized_seconds_per_sample"]),
            "time_std": float(row["std_amortized_seconds_per_sample"]),
            "N": int(row["N"]),
            "P": int(row["P"]),
            "L": int(row["L"]),
            "batch_size": int(row["batch_size"]),
            "hardware": row["hardware"],
            "precision": row["precision"],
        }
        for row in rows
    ]


def normalize_adaptive(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "schedule_type": "adaptive",
            "label": f"τ=log({float(row['tau_multiplier']):g})",
            "schedule_value": row["threshold_label"],
            "quality": float(row["MAUVE"]),
            "time": float(row["mean_amortized_seconds_per_sample"]),
            "time_std": float(row["std_amortized_seconds_per_sample"]),
            "N": int(row["N"]),
            "P": int(row["P"]),
            "L": int(row["L"]),
            "batch_size": int(row["batch_size"]),
            "hardware": row["hardware"],
            "precision": row["precision"],
        }
        for row in rows
    ]


def validate_comparison(points: Sequence[dict[str, Any]]) -> None:
    for field in ("N", "P", "L", "batch_size", "hardware", "precision"):
        values = {point[field] for point in points}
        if len(values) != 1:
            raise ValueError(
                f"Cannot compare runs because {field} differs: {sorted(values)}"
            )
    for point in points:
        if (
            not math.isfinite(point["quality"])
            or not 0.0 <= point["quality"] <= 1.0
        ):
            raise ValueError(f"Invalid MAUVE value for {point['label']}.")
        if not math.isfinite(point["time"]) or point["time"] <= 0.0:
            raise ValueError(f"Invalid timing value for {point['label']}.")


def pareto_frontier(
    points: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    frontier = []
    for candidate in points:
        dominated = any(
            other["quality"] >= candidate["quality"]
            and other["time"] <= candidate["time"]
            and (
                other["quality"] > candidate["quality"]
                or other["time"] < candidate["time"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda point: point["quality"])


def write_combined_csv(
    path: Path,
    points: Sequence[dict[str, Any]],
    frontier: Sequence[dict[str, Any]],
) -> None:
    frontier_keys = {
        (point["schedule_type"], str(point["schedule_value"]))
        for point in frontier
    }
    fields = (
        "schedule_type",
        "label",
        "schedule_value",
        "MAUVE",
        "amortized_seconds_per_sample",
        "std_seconds_per_sample",
        "pareto_nondominated",
        "N",
        "P",
        "L",
        "batch_size",
        "hardware",
        "precision",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "schedule_type": point["schedule_type"],
                    "label": point["label"],
                    "schedule_value": point["schedule_value"],
                    "MAUVE": point["quality"],
                    "amortized_seconds_per_sample": point["time"],
                    "std_seconds_per_sample": point["time_std"],
                    "pareto_nondominated": (
                        point["schedule_type"],
                        str(point["schedule_value"]),
                    )
                    in frontier_keys,
                    "N": point["N"],
                    "P": point["P"],
                    "L": point["L"],
                    "batch_size": point["batch_size"],
                    "hardware": point["hardware"],
                    "precision": point["precision"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    fixed = normalize_fixed(
        read_csv(args.fixed_summary.expanduser().resolve())
    )
    adaptive = normalize_adaptive(
        read_csv(args.adaptive_summary.expanduser().resolve())
    )
    points = fixed + adaptive
    validate_comparison(points)
    frontier = pareto_frontier(points)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_combined_csv(output_dir / "time_vs_quality_points.csv", points, frontier)

    fig, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    fixed_by_quality = sorted(fixed, key=lambda point: point["quality"])
    axis.plot(
        [point["quality"] for point in fixed_by_quality],
        [point["time"] for point in fixed_by_quality],
        color="#2563eb",
        linewidth=1.8,
        alpha=0.75,
        label="Fixed schedules (connected by MAUVE)",
        zorder=2,
    )
    axis.errorbar(
        [point["quality"] for point in fixed],
        [point["time"] for point in fixed],
        yerr=[point["time_std"] for point in fixed],
        fmt="o",
        markersize=7,
        color="#1d4ed8",
        capsize=3,
        linestyle="none",
        zorder=3,
    )
    axis.errorbar(
        [point["quality"] for point in adaptive],
        [point["time"] for point in adaptive],
        yerr=[point["time_std"] for point in adaptive],
        fmt="D",
        markersize=7,
        color="#dc2626",
        capsize=3,
        linestyle="none",
        label="Adaptive schedules",
        zorder=4,
    )
    if len(frontier) > 1:
        axis.plot(
            [point["quality"] for point in frontier],
            [point["time"] for point in frontier],
            color="#111827",
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            label="Nondominated frontier",
            zorder=1,
        )

    for point in points:
        axis.annotate(
            point["label"],
            (point["quality"], point["time"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
        )
    axis.set_xlabel("MAUVE (higher is better)")
    axis.set_ylabel("Synchronized seconds per completed sample (lower is better)")
    axis.set_title(
        "Dream-7B Time–Quality Comparison\n"
        f"N={points[0]['N']}, P={points[0]['P']}, L={points[0]['L']}"
    )
    if args.log_time_axis:
        axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)

    png = output_dir / "time_vs_quality.png"
    pdf = output_dir / "time_vs_quality.pdf"
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"[done] comparison plot: {png}")
    print(f"[done] vector plot:     {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
