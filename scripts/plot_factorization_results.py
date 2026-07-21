#!/usr/bin/env python3
"""Plot exhaustive-subset factorization gaps from one Dream rollout.

The runner writes one row for every ``(step_index, subset_mask)`` pair. This
plotter includes singleton subsets (whose factorization gap is zero by
construction) and keeps two complementary analysis levels:

1. **By subset size k**: pool all confidence-rank subsets of the same size.
   This answers how approximation error scales with the number of tokens.
2. **By exact confidence-rank subset**: keep identities such as ``{1,3}`` or
   ``{4,5,6}`` separate across rollout steps. This answers whether particular
   confidence-rank patterns are systematically easier or harder.

Pairwise subsets retain their dedicated summary and appear in the consolidated
boxplot figure, but no heatmaps are generated. An exact subset is a *rank
pattern*, not a fixed token identity: ``{1,3}``
means the most-confident and third-most-confident frozen proposals at each
state. Token IDs and canvas positions may change from step to step.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Command-line and input-path handling
# -----------------------------------------------------------------------------

# Define the plotting interface while keeping the original one-command usage.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        type=Path,
        help=(
            "Run directory containing estimates.jsonl and steps.jsonl, "
            "or the path to estimates.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to <run>/plots.",
    )
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--top-subsets",
        type=int,
        default=20,
        help=(
            "Number of exact subsets shown in the ranked median-error figure. "
            "The CSV still contains every subset."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    args = parser.parse_args(argv)

    if args.bins < 1:
        parser.error("--bins must be at least 1.")
    if args.dpi < 1:
        parser.error("--dpi must be at least 1.")
    if args.top_subsets < 1:
        parser.error("--top-subsets must be at least 1.")
    return args


# Resolve either a run directory or a direct estimates.jsonl path.
def resolve_paths(run: Path) -> tuple[Path, Path, Path, Path | None]:
    run = run.expanduser().resolve()

    if run.is_dir():
        run_dir = run
        estimates_path = run_dir / "estimates.jsonl"
    else:
        estimates_path = run
        run_dir = estimates_path.parent

    steps_path = run_dir / "steps.jsonl"
    metadata_path = run_dir / "metadata.json"

    for path in (estimates_path, steps_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    return (
        run_dir,
        estimates_path,
        steps_path,
        metadata_path if metadata_path.exists() else None,
    )


# Read JSONL explicitly so nested lists and dictionaries retain their shape.
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            rows.append(value)

    if not rows:
        raise ValueError(f"No JSON records were found in {path}.")
    return rows


# Read optional metadata for exact expected subset counts and report context.
def load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


# -----------------------------------------------------------------------------
# Schema normalization
# -----------------------------------------------------------------------------

# Convert list-like subset ranks into one stable tuple representation.
def normalize_subset_ranks(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        ranks = tuple(int(item) for item in value)
    elif isinstance(value, list):
        ranks = tuple(int(item) for item in value)
    elif isinstance(value, str):
        stripped = value.strip().strip("{}[]()")
        ranks = (
            tuple(int(item.strip()) for item in stripped.split(",") if item.strip())
            if stripped
            else ()
        )
    else:
        raise TypeError(f"Unsupported subset_ranks value: {value!r}")

    if not ranks:
        raise ValueError("subset_ranks must not be empty.")
    if any(rank < 1 for rank in ranks):
        raise ValueError(f"subset ranks must be positive: {ranks}")
    if tuple(sorted(ranks)) != ranks:
        raise ValueError(f"subset ranks must be increasing: {ranks}")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"subset ranks must be unique: {ranks}")
    return ranks


# Encode one-based confidence ranks in the same bit-mask convention as runner.
def subset_mask_from_ranks(ranks: Sequence[int]) -> int:
    mask = 0
    for rank in ranks:
        mask |= 1 << (int(rank) - 1)
    return mask


# Make the concise human-readable identity used on axes and in CSV reports.
def subset_label(ranks: Sequence[int]) -> str:
    return "{" + ",".join(str(int(rank)) for rank in ranks) + "}"


# Load schema-v2 exhaustive rows and provide a narrow compatibility path for
# old prefix-only rows by interpreting k as the prefix {1,...,k}.
def load_estimates(path: Path) -> pd.DataFrame:
    records = read_jsonl(path)
    frame = pd.DataFrame(records)

    required_metrics = {
        "step_index",
        "difference_joint_minus_product",
        "log_gap_joint_minus_product",
    }
    missing_metrics = required_metrics - set(frame.columns)
    if missing_metrics:
        raise ValueError(
            f"estimates.jsonl is missing columns: {sorted(missing_metrics)}"
        )

    frame = frame.copy()

    # Current schema: exact subset identity is already explicit.
    if "subset_ranks" in frame.columns:
        frame["subset_ranks"] = frame["subset_ranks"].map(
            normalize_subset_ranks
        )
    # Legacy schema: one nested confidence prefix per k.
    elif "k" in frame.columns:
        frame["subset_ranks"] = frame["k"].map(
            lambda value: tuple(range(1, int(value) + 1))
        )
    else:
        raise ValueError(
            "estimates.jsonl must contain subset_ranks (schema 2) or k "
            "(legacy prefix schema)."
        )

    computed_sizes = frame["subset_ranks"].map(len).astype(int)
    if "subset_size" in frame.columns:
        recorded_sizes = frame["subset_size"].astype(int)
        if not recorded_sizes.equals(computed_sizes):
            raise ValueError("subset_size disagrees with len(subset_ranks).")
    frame["subset_size"] = computed_sizes

    # Retain k as an explicit compatibility alias, but use subset_size
    # internally so the meaning remains unambiguous.
    if "k" in frame.columns:
        recorded_k = frame["k"].astype(int)
        if not recorded_k.equals(frame["subset_size"]):
            raise ValueError("k disagrees with subset_size.")
    frame["k"] = frame["subset_size"]

    computed_masks = frame["subset_ranks"].map(subset_mask_from_ranks).astype(int)
    if "subset_mask" in frame.columns:
        recorded_masks = frame["subset_mask"].astype(int)
        if not recorded_masks.equals(computed_masks):
            raise ValueError("subset_mask disagrees with subset_ranks.")
    frame["subset_mask"] = computed_masks

    computed_labels = frame["subset_ranks"].map(subset_label)
    if "subset_label" in frame.columns:
        recorded_labels = frame["subset_label"].astype(str)
        if not recorded_labels.equals(computed_labels):
            raise ValueError("subset_label disagrees with subset_ranks.")
    frame["subset_label"] = computed_labels

    frame["step_index"] = frame["step_index"].astype(int)
    frame["difference_joint_minus_product"] = pd.to_numeric(
        frame["difference_joint_minus_product"], errors="raise"
    )
    frame["log_gap_joint_minus_product"] = pd.to_numeric(
        frame["log_gap_joint_minus_product"], errors="raise"
    )
    frame["absolute_difference"] = frame[
        "difference_joint_minus_product"
    ].abs()
    frame["absolute_log_gap"] = frame[
        "log_gap_joint_minus_product"
    ].abs()

    numeric_columns = [
        "difference_joint_minus_product",
        "log_gap_joint_minus_product",
        "absolute_difference",
        "absolute_log_gap",
    ]
    numeric_values = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Gap columns contain NaN or infinite values.")

    duplicate_key = ["step_index", "subset_mask"]
    if frame.duplicated(duplicate_key).any():
        duplicates = frame.loc[
            frame.duplicated(duplicate_key, keep=False),
            duplicate_key + ["subset_label"],
        ]
        raise ValueError(
            "Duplicate (step_index, subset_mask) rows were found:\n"
            + duplicates.to_string(index=False)
        )

    return frame.sort_values(
        ["subset_size", "subset_mask", "step_index"]
    ).reset_index(drop=True)


# Flatten the nested committed-token record while preserving exact decoded
# canvas snapshots written by the current runner.
def load_steps(path: Path) -> pd.DataFrame:
    records = read_jsonl(path)
    rows: list[dict[str, Any]] = []

    for record in records:
        committed = record.get("committed")
        if not isinstance(committed, dict):
            committed = {}

        rows.append(
            {
                "step_index": int(record["step_index"]),
                "remaining_mask_count_before": int(
                    record.get("remaining_mask_count_before", -1)
                ),
                "remaining_mask_count_after": int(
                    record.get("remaining_mask_count_after", -1)
                ),
                "committed_canvas_position": int(
                    record.get(
                        "committed_canvas_position",
                        committed.get("canvas_position", -1),
                    )
                ),
                "committed_token_text": record.get(
                    "committed_token_text_repr",
                    committed.get(
                        "token_text_repr",
                        committed.get("token_text", ""),
                    ),
                ),
                "committed_confidence": float(
                    record.get(
                        "committed_confidence",
                        committed.get("probability", math.nan),
                    )
                ),
                "canvas_after_without_specials": record.get(
                    "canvas_after_without_specials", ""
                ),
                "canvas_after_with_masks": record.get(
                    "canvas_after_with_masks", ""
                ),
            }
        )

    frame = pd.DataFrame(rows).sort_values("step_index").reset_index(drop=True)
    if frame["step_index"].duplicated().any():
        raise ValueError("Duplicate step_index rows were found in steps.jsonl.")
    if not np.isfinite(frame["committed_confidence"].to_numpy(dtype=float)).all():
        raise ValueError("Committed confidence contains NaN or infinity.")
    return frame


# -----------------------------------------------------------------------------
# Cross-file and combinatorial audits
# -----------------------------------------------------------------------------

# Extract the declared exhaustive experiment shape from metadata when present.
def metadata_experiment_shape(
    metadata: dict[str, Any],
) -> tuple[int | None, tuple[int, ...] | None]:
    experiment = metadata.get("experiment")
    if not isinstance(experiment, dict):
        return None, None

    max_k_value = experiment.get("max_k")
    subset_sizes_value = experiment.get("subset_sizes")
    max_k = int(max_k_value) if max_k_value is not None else None
    subset_sizes = (
        tuple(int(value) for value in subset_sizes_value)
        if isinstance(subset_sizes_value, list)
        else None
    )
    return max_k, subset_sizes


# Enumerate the expected one-based rank identities for a declared experiment.
def expected_subsets(
    max_k: int,
    subset_sizes: Sequence[int],
) -> set[tuple[int, ...]]:
    return {
        subset
        for size in subset_sizes
        for subset in combinations(range(1, max_k + 1), int(size))
    }


# Verify that every requested exact subset appears once at every rollout state,
# while avoiding the obsolete one-row-per-(step,k) assumption.
def audit(
    estimates: pd.DataFrame,
    steps: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    expected_step_indices = set(steps["step_index"].astype(int))
    observed_step_indices = set(estimates["step_index"].astype(int))
    if observed_step_indices != expected_step_indices:
        raise ValueError(
            "estimates.jsonl and steps.jsonl contain different step indices."
        )

    # Every exact rank subset must have one observation for every canonical
    # step. This is the correct schema-v2 replacement for one row per k.
    for subset, group in estimates.groupby("subset_ranks", sort=False):
        observed = set(group["step_index"].astype(int))
        if observed != expected_step_indices:
            raise ValueError(
                f"Subset {subset_label(subset)} does not occur exactly once "
                "at every step."
            )

    max_k_meta, subset_sizes_meta = metadata_experiment_shape(metadata)
    if max_k_meta is not None and subset_sizes_meta is not None:
        declared = expected_subsets(max_k_meta, subset_sizes_meta)
        observed = set(estimates["subset_ranks"])
        if observed != declared:
            missing = sorted(declared - observed, key=lambda x: (len(x), x))
            extra = sorted(observed - declared, key=lambda x: (len(x), x))
            raise ValueError(
                "Observed exact subsets do not match metadata. "
                f"missing={missing}, extra={extra}"
            )

        expected_rows = len(expected_step_indices) * len(declared)
        if len(estimates) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} estimate rows from metadata, "
                f"found {len(estimates)}."
            )

    # Validate the canonical one-token-per-step transition when mask counts are
    # present. Legacy files may use -1 sentinels and are skipped here.
    valid_mask_rows = steps[
        (steps["remaining_mask_count_before"] >= 0)
        & (steps["remaining_mask_count_after"] >= 0)
    ]
    if not valid_mask_rows.empty:
        decrements = (
            valid_mask_rows["remaining_mask_count_before"]
            - valid_mask_rows["remaining_mask_count_after"]
        )
        if not (decrements == 1).all():
            raise ValueError("At least one rollout step did not remove one mask.")

    # Singleton products must be identical by construction: P_{i}=Q_{i}.
    singletons = estimates.loc[estimates["subset_size"] == 1]
    if not singletons.empty:
        tolerance = 1e-10
        if (singletons["absolute_log_gap"] > tolerance).any():
            raise ValueError("A singleton subset has a nonzero log gap.")


# -----------------------------------------------------------------------------
# Descriptive statistics
# -----------------------------------------------------------------------------

# Return the same descriptive statistics for each scalar metric.
def describe_metric(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot describe an empty metric array.")
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_max": float(np.max(values)),
    }


# Pool every exact subset of the same cardinality, including singletons.
def make_size_summary(estimates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for subset_size, group in estimates.groupby("subset_size", sort=True):
        row: dict[str, Any] = {
            "k": int(subset_size),
            "subset_size": int(subset_size),
            "count": int(len(group)),
            "unique_subsets": int(group["subset_mask"].nunique()),
            "rollout_steps": int(group["step_index"].nunique()),
            "fraction_joint_greater_than_product": float(
                np.mean(group["log_gap_joint_minus_product"].to_numpy(float) > 0)
            ),
        }
        row.update(
            describe_metric(
                group["absolute_difference"].to_numpy(float),
                "absolute_raw_gap",
            )
        )
        row.update(
            describe_metric(
                group["absolute_log_gap"].to_numpy(float),
                "absolute_log_gap",
            )
        )
        row.update(
            describe_metric(
                group["log_gap_joint_minus_product"].to_numpy(float),
                "signed_log_gap",
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


# Keep every confidence-rank identity separate across denoising states.
def make_exact_subset_summary(estimates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for subset, group in estimates.groupby("subset_ranks", sort=False):
        subset_tuple = tuple(int(rank) for rank in subset)
        signed_log = group["log_gap_joint_minus_product"].to_numpy(float)
        row: dict[str, Any] = {
            "subset_label": subset_label(subset_tuple),
            "subset_ranks": ",".join(str(rank) for rank in subset_tuple),
            "subset_mask": subset_mask_from_ranks(subset_tuple),
            "subset_size": len(subset_tuple),
            "count": int(len(group)),
            "rank_min": min(subset_tuple),
            "rank_max": max(subset_tuple),
            "rank_span": max(subset_tuple) - min(subset_tuple),
            "mean_rank": float(np.mean(subset_tuple)),
            "contains_rank_1": int(1 in subset_tuple),
            "fraction_joint_greater_than_product": float(np.mean(signed_log > 0)),
        }
        row.update(
            describe_metric(
                group["absolute_difference"].to_numpy(float),
                "absolute_raw_gap",
            )
        )
        row.update(
            describe_metric(
                group["absolute_log_gap"].to_numpy(float),
                "absolute_log_gap",
            )
        )
        row.update(describe_metric(signed_log, "signed_log_gap"))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["subset_size", "subset_mask"]
    ).reset_index(drop=True)


# Add pair-specific rank columns to the exact-subset statistics.
def make_pairwise_summary(exact_summary: pd.DataFrame) -> pd.DataFrame:
    pairwise = exact_summary.loc[exact_summary["subset_size"] == 2].copy()
    if pairwise.empty:
        return pairwise

    parsed = pairwise["subset_ranks"].map(
        lambda value: tuple(int(item) for item in str(value).split(","))
    )
    pairwise["earlier_confidence_rank"] = parsed.map(lambda ranks: ranks[0])
    pairwise["later_confidence_rank"] = parsed.map(lambda ranks: ranks[1])
    pairwise["rank_distance"] = (
        pairwise["later_confidence_rank"]
        - pairwise["earlier_confidence_rank"]
    )
    return pairwise.sort_values(
        ["earlier_confidence_rank", "later_confidence_rank"]
    ).reset_index(drop=True)


# Aggregate the within-step distribution over exact subsets before drawing a
# by-k trajectory. Median and IQR avoid plotting many duplicate x-values.
def make_size_trajectory(estimates: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = estimates.groupby(["subset_size", "step_index"])[column]
    trajectory = grouped.agg(
        median="median",
        mean="mean",
        minimum="min",
        maximum="max",
        count="count",
    ).reset_index()
    quantiles = grouped.quantile([0.25, 0.75]).unstack(level=-1).reset_index()
    quantiles = quantiles.rename(columns={0.25: "q25", 0.75: "q75"})
    return trajectory.merge(
        quantiles,
        on=["subset_size", "step_index"],
        validate="one_to_one",
    )


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------

# Draw a labeled boxplot across Matplotlib versions. New releases renamed
# ``labels`` to ``tick_labels``; the fallback keeps older cluster installs usable.
def draw_labeled_boxplot(
    axis: plt.Axes,
    values: Sequence[np.ndarray],
    labels: Sequence[str],
) -> None:
    style = {
        "showmeans": True,
        "medianprops": {"color": "tab:orange", "linewidth": 1.6},
        "meanprops": {
            "marker": "^",
            "markerfacecolor": "tab:green",
            "markeredgecolor": "tab:green",
            "markersize": 6,
        },
        "flierprops": {
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "markersize": 4,
        },
    }
    try:
        axis.boxplot(values, tick_labels=labels, **style)
    except TypeError:
        axis.boxplot(values, labels=labels, **style)


def boxplot_legend_handles() -> list[object]:
    """Return one shared visual key for every boxplot panel."""

    return [
        Patch(
            facecolor="white",
            edgecolor="black",
            label="Box: middle 50% (IQR)",
        ),
        Line2D(
            [0], [0], color="tab:orange", linewidth=1.6, label="Median"
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="tab:green",
            markeredgecolor="tab:green",
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1,
            label="Whiskers: within 1.5×IQR",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="black",
            label="Outlier beyond whiskers",
        ),
    ]


# Save figures consistently and close them so many plots do not accumulate.
def save_figure(
    figure: plt.Figure,
    path: Path,
    *,
    dpi: int,
    show: bool,
    layout_rect: tuple[float, float, float, float] | None = None,
) -> None:
    figure.tight_layout(rect=layout_rect)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(figure)


# Draw raw-gap and log-gap histograms together for each subset cardinality.
def plot_histograms_by_size(
    estimates: pd.DataFrame,
    *,
    output_path: Path,
    bins: int,
    dpi: int,
    show: bool,
) -> None:
    sizes = sorted(estimates["subset_size"].unique())
    metrics = (
        ("absolute_log_gap", "|log(P_S / Q_S)|", "Absolute log gap"),
        ("absolute_difference", "|P_S - Q_S|", "Absolute raw gap"),
    )
    figure, axes = plt.subplots(
        len(sizes),
        len(metrics),
        figsize=(14.0, max(3.4, 3.2 * len(sizes))),
        squeeze=False,
    )

    for row_index, subset_size in enumerate(sizes):
        size_rows = estimates.loc[estimates["subset_size"] == subset_size]
        for column_index, (column, xlabel, metric_title) in enumerate(metrics):
            axis = axes[row_index, column_index]
            values = size_rows[column].to_numpy(float)
            axis.hist(values, bins=bins, edgecolor="black")
            axis.axvline(
                np.mean(values),
                color="tab:orange",
                linestyle="--",
                linewidth=1.2,
            )
            axis.axvline(
                np.median(values),
                color="tab:green",
                linestyle=":",
                linewidth=1.5,
            )
            axis.set_title(
                f"{metric_title}, k={int(subset_size)}, "
                f"rows={len(values)}, subsets={size_rows['subset_mask'].nunique()}"
            )
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Count")
            axis.text(
                0.98,
                0.94,
                f"mean={np.mean(values):.4g}\nmedian={np.median(values):.4g}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    figure.legend(
        handles=[
            Line2D(
                [0], [0], color="tab:orange", linestyle="--", label="Mean"
            ),
            Line2D(
                [0], [0], color="tab:green", linestyle=":", label="Median"
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    save_figure(
        figure,
        output_path,
        dpi=dpi,
        show=show,
        layout_rect=(0.0, 0.04, 1.0, 1.0),
    )


# Put pooled and exact-subset boxplots into one chart-type figure. The exact
# k=2 panel is the pairwise boxplot.
def plot_all_boxplots(
    estimates: pd.DataFrame,
    *,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    sizes = sorted(estimates["subset_size"].unique())
    metrics = (
        (
            "absolute_log_gap",
            "|log(P_S / Q_S)|",
            "Absolute log-gap distribution",
        ),
        ("absolute_difference", "|P_S - Q_S|", "Absolute raw-gap distribution"),
    )
    maximum_exact_subsets = int(
        estimates.groupby("subset_size")["subset_mask"].nunique().max()
    )
    figure_width = max(16.0, min(24.0, 0.55 * maximum_exact_subsets + 8.0))
    figure = plt.figure(figsize=(figure_width, 5.0 + 4.6 * len(sizes)))
    grid = figure.add_gridspec(
        len(sizes) + 1,
        2,
        height_ratios=[1.0] + [1.15] * len(sizes),
    )

    for column_index, (column, ylabel, title) in enumerate(metrics):
        axis = figure.add_subplot(grid[0, column_index])
        values = [
            estimates.loc[
                estimates["subset_size"] == size, column
            ].to_numpy(float)
            for size in sizes
        ]
        draw_labeled_boxplot(
            axis,
            values,
            [str(int(size)) for size in sizes],
        )
        axis.set_xlabel("Subset size k")
        axis.set_ylabel(ylabel)
        axis.set_title(title + " pooled by subset size")

    for row_index, (subset_size, size_group) in enumerate(
        estimates.groupby("subset_size", sort=True),
        start=1,
    ):
        axis = figure.add_subplot(grid[row_index, :])
        labels: list[str] = []
        values: list[np.ndarray] = []
        for ranks, subset_group in size_group.groupby("subset_ranks", sort=False):
            labels.append(subset_label(ranks))
            values.append(subset_group["absolute_log_gap"].to_numpy(float))

        draw_labeled_boxplot(axis, values, labels)
        axis.set_xlabel("Exact confidence-rank subset S")
        axis.set_ylabel("|log(P_S / Q_S)|")
        qualifier = " (pairwise subsets)" if int(subset_size) == 2 else ""
        axis.set_title(
            f"Exact-subset absolute log gaps, k={int(subset_size)}{qualifier}"
        )
        axis.tick_params(axis="x", rotation=55)

    figure.legend(
        handles=boxplot_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    save_figure(
        figure,
        output_path,
        dpi=dpi,
        show=show,
        layout_rect=(0.0, 0.055, 1.0, 1.0),
    )


# Plot the median and interquartile subset error at each rollout state for each
# k. This avoids connecting arbitrary exact subsets as if there were one row.
def plot_size_trajectory(
    estimates: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    trajectory = make_size_trajectory(estimates, column)
    figure, axis = plt.subplots(figsize=(10.5, 5.8))

    for subset_size, group in trajectory.groupby("subset_size", sort=True):
        group = group.sort_values("step_index")
        x = group["step_index"].to_numpy(float)
        median = group["median"].to_numpy(float)
        q25 = group["q25"].to_numpy(float)
        q75 = group["q75"].to_numpy(float)
        line = axis.plot(
            x,
            median,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"k={int(subset_size)}",
        )[0]
        axis.fill_between(x, q25, q75, alpha=0.18, color=line.get_color())

    axis.set_xlabel("Rollout step t")
    axis.set_ylabel(ylabel)
    axis.set_title(title + " (median and IQR across exact subsets)")
    axis.legend(title="Lines: median; bands: IQR")
    save_figure(figure, output_path, dpi=dpi, show=show)


# Plot pooled empirical CDFs by subset cardinality.
def plot_ecdf_by_size(
    estimates: pd.DataFrame,
    *,
    column: str,
    xlabel: str,
    title: str,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))

    for subset_size, group in estimates.groupby("subset_size", sort=True):
        values = np.sort(group[column].to_numpy(float))
        cumulative = np.arange(1, len(values) + 1) / len(values)
        axis.step(
            values,
            cumulative,
            where="post",
            label=f"k={int(subset_size)}",
        )

    axis.set_xlabel(xlabel)
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title(title)
    axis.set_ylim(0.0, 1.0)
    axis.legend(title="Subset size")
    save_figure(figure, output_path, dpi=dpi, show=show)


# Rank exact subset identities by their median error across the trajectory.
def plot_top_exact_subsets(
    exact_summary: pd.DataFrame,
    *,
    top_n: int,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    ranked = exact_summary.nlargest(
        min(top_n, len(exact_summary)),
        "absolute_log_gap_median",
    ).sort_values("absolute_log_gap_median")

    figure, axis = plt.subplots(
        figsize=(9.0, max(5.5, 0.34 * len(ranked) + 2.0))
    )
    labels = [
        f"{label} (k={size})"
        for label, size in zip(ranked["subset_label"], ranked["subset_size"])
    ]
    axis.barh(labels, ranked["absolute_log_gap_median"].to_numpy(float))
    axis.set_xlabel("Median |log(P_S / Q_S)| across rollout steps")
    axis.set_ylabel("Exact confidence-rank subset")
    axis.set_title(f"Top {len(ranked)} exact subsets by median absolute log gap")
    save_figure(figure, output_path, dpi=dpi, show=show)


# -----------------------------------------------------------------------------
# Denoising trajectory and report output
# -----------------------------------------------------------------------------

# Rename the flattened step fields into a compact human-facing table.
def make_denoising_trajectory(steps: pd.DataFrame) -> pd.DataFrame:
    return steps[
        [
            "step_index",
            "committed_canvas_position",
            "committed_token_text",
            "committed_confidence",
            "canvas_after_without_specials",
            "canvas_after_with_masks",
        ]
    ].rename(
        columns={
            "step_index": "step",
            "committed_canvas_position": "canvas_position",
            "committed_token_text": "committed_token",
            "committed_confidence": "confidence",
            "canvas_after_without_specials": "visible_sequence_after_step",
            "canvas_after_with_masks": "full_canvas_after_step",
        }
    )


# Preserve whitespace-sensitive decoded sequences in a plain-text audit file.
def write_text_trajectory(trajectory: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in trajectory.itertuples(index=False):
            handle.write(
                f"step={int(row.step):02d}  "
                f"position={int(row.canvas_position):02d}  "
                f"token={row.committed_token!r}  "
                f"confidence={float(row.confidence):.8f}\n"
            )
            handle.write(f"visible: {row.visible_sequence_after_step}\n")
            if row.full_canvas_after_step:
                handle.write(f"canvas:  {row.full_canvas_after_step}\n")
            handle.write("\n")


# Render a DataFrame as a multipage LaTeX table.
def dataframe_to_latex(
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
) -> str:
    return frame.to_latex(
        index=False,
        escape=True,
        longtable=True,
        float_format=lambda value: f"{value:.6g}",
        caption=caption,
        label=label,
    )


# Select compact columns for report tables while leaving full detail in CSV.
def compact_size_summary(size_summary: pd.DataFrame) -> pd.DataFrame:
    return size_summary[
        [
            "k",
            "count",
            "unique_subsets",
            "absolute_log_gap_mean",
            "absolute_log_gap_median",
            "absolute_log_gap_q25",
            "absolute_log_gap_q75",
            "absolute_log_gap_max",
            "fraction_joint_greater_than_product",
        ]
    ].rename(
        columns={
            "count": "n",
            "unique_subsets": "subsets",
            "absolute_log_gap_mean": "mean_abs_log_gap",
            "absolute_log_gap_median": "median_abs_log_gap",
            "absolute_log_gap_q25": "q25",
            "absolute_log_gap_q75": "q75",
            "absolute_log_gap_max": "max_abs_log_gap",
            "fraction_joint_greater_than_product": "fraction_P_gt_Q",
        }
    )


# Write a standalone LaTeX report containing tables and generated figures.
def write_report(
    *,
    size_summary: pd.DataFrame,
    exact_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    trajectory: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_path = output_dir / "report.tex"
    top_exact = exact_summary.nlargest(
        min(15, len(exact_summary)),
        "absolute_log_gap_median",
    )[
        [
            "subset_label",
            "subset_size",
            "count",
            "absolute_log_gap_median",
            "absolute_log_gap_q25",
            "absolute_log_gap_q75",
            "signed_log_gap_median",
            "fraction_joint_greater_than_product",
        ]
    ].rename(
        columns={
            "subset_label": "subset",
            "subset_size": "k",
            "count": "n",
            "absolute_log_gap_median": "median_abs_log_gap",
            "absolute_log_gap_q25": "q25",
            "absolute_log_gap_q75": "q75",
            "signed_log_gap_median": "median_signed_log_gap",
            "fraction_joint_greater_than_product": "fraction_P_gt_Q",
        }
    )

    pair_table = (
        pairwise_summary[
            [
                "subset_label",
                "rank_distance",
                "count",
                "absolute_log_gap_median",
                "absolute_log_gap_q25",
                "absolute_log_gap_q75",
                "signed_log_gap_median",
                "fraction_joint_greater_than_product",
            ]
        ]
        .sort_values("absolute_log_gap_median", ascending=False)
        .rename(
            columns={
                "subset_label": "pair",
                "rank_distance": "rank_distance",
                "count": "n",
                "absolute_log_gap_median": "median_abs_log_gap",
                "absolute_log_gap_q25": "q25",
                "absolute_log_gap_q75": "q75",
                "signed_log_gap_median": "median_signed_log_gap",
                "fraction_joint_greater_than_product": "fraction_P_gt_Q",
            }
        )
        if not pairwise_summary.empty
        else pd.DataFrame()
    )

    report_trajectory = trajectory[
        [
            "step",
            "canvas_position",
            "committed_token",
            "confidence",
        ]
    ].copy()

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\\documentclass[11pt]{article}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{longtable}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{float}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\graphicspath{{./}}\n"
            "\\title{Exhaustive-subset Factorization-gap Rollout Report}\n"
            "\\date{}\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
        )
        handle.write(
            "All analyses include singleton subsets. Their factorization gap "
            "should be zero by construction and provides a useful reference. "
            "Magnitude plots use absolute gaps; "
            "signed log-gap statistics remain in the CSV files and tables.\n\n"
        )
        handle.write("\\section{How to read the aggregation levels}\n")
        handle.write(
            "\\textbf{By $k$} pools every exact confidence-rank subset with the same "
            "cardinality. It measures how error scales with the number of "
            "tokens, but it can hide rank-pattern heterogeneity.\\par\n"
        )
        handle.write(
            "\\textbf{By exact subset} keeps rank identities such as "
            "\\{1,3\\} separate across steps. \\{1,3\\} means the first and third most "
            "confident proposals at each state; it does not track fixed token "
            "IDs or fixed canvas positions.\\par\n\n"
        )
        handle.write(
            "The rollout steps are serially correlated observations from one "
            "deterministic trajectory, not independent samples.\n\n"
        )

        handle.write("\\section{Visual key}\n")
        handle.write(
            "In every boxplot, the box spans the interquartile range (IQR), "
            "from the 25th to the 75th percentile. The orange line is the "
            "median, the green triangle is the mean, whiskers extend to the "
            "most extreme observations within $1.5\\times$ IQR, and hollow "
            "circles are observations beyond the whiskers. In histograms, "
            "the dashed orange line marks the mean and the dotted green line "
            "marks the median. In the trajectory plot, each line is a "
            "within-step median and its shaded band is the IQR.\\par\n\n"
        )

        handle.write("\\section{Summary by subset size}\n")
        handle.write(
            dataframe_to_latex(
                compact_size_summary(size_summary),
                caption="Summary statistics by subset size, including singletons.",
                label="tab:size-summary",
            )
        )
        handle.write("\n\\section{Exact subsets with the largest median error}\n")
        handle.write(
            dataframe_to_latex(
                top_exact,
                caption="Exact subsets ranked by median absolute log gap.",
                label="tab:exact-summary",
            )
        )

        if not pair_table.empty:
            handle.write("\n\\section{Pairwise confidence-rank summary}\n")
            handle.write(
                dataframe_to_latex(
                    pair_table,
                    caption="Summary statistics for confidence-rank pairs.",
                    label="tab:pairwise-summary",
                )
            )

        handle.write("\n\\section{Figures}\n")
        figure_names = [
            "factorization_gap_histograms.png",
            "factorization_gap_boxplots.png",
            "absolute_log_gap_trajectory.png",
            "absolute_log_gap_ecdf.png",
            "top_exact_subsets_by_median_absolute_log_gap.png",
        ]

        for filename in figure_names:
            caption = filename.replace("_", "\\_")
            handle.write(
                "\\begin{figure}[H]\n"
                "\\centering\n"
                f"\\includegraphics[width=0.98\\textwidth]{{{filename}}}\n"
                f"\\caption{{{caption}}}\n"
                "\\end{figure}\n\n"
            )

        handle.write("\\section{Denoising trajectory}\n")
        handle.write(
            dataframe_to_latex(
                report_trajectory,
                caption="Denoising trajectory.",
                label="tab:trajectory",
            )
        )
        handle.write("\n\\end{document}\n")

    print(f"Saved: {report_path}")


# Remove artifacts generated by older versions so reruns have one clear
# output contract. Only known plotter-owned files are touched.
def remove_retired_artifacts(output_dir: Path) -> None:
    retired_names = (
        "report.md",
        "absolute_log_gap_histograms.png",
        "absolute_log_gap_boxplot.png",
        "absolute_raw_gap_histograms.png",
        "absolute_raw_gap_boxplot.png",
        "pairwise_absolute_log_gap_boxplot.png",
        "pairwise_median_absolute_log_gap_heatmap.png",
        "pairwise_median_signed_log_gap_heatmap.png",
    )
    for name in retired_names:
        path = output_dir / name
        if path.is_file():
            path.unlink()
            print(f"Removed retired artifact: {path}")
    for path in output_dir.glob(
        "exact_subset_absolute_log_gap_boxplot_k*.png"
    ):
        if path.is_file():
            path.unlink()
            print(f"Removed retired artifact: {path}")


# -----------------------------------------------------------------------------
# Executable orchestration
# -----------------------------------------------------------------------------

# Load, audit, summarize, plot, and write all machine- and human-readable files.
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir, estimates_path, steps_path, metadata_path = resolve_paths(args.run)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_retired_artifacts(output_dir)

    estimates = load_estimates(estimates_path)
    steps = load_steps(steps_path)
    metadata = load_metadata(metadata_path)
    audit(estimates, steps, metadata)

    analysis_estimates = estimates.copy()

    size_summary = make_size_summary(analysis_estimates)
    exact_summary = make_exact_subset_summary(analysis_estimates)
    pairwise_summary = make_pairwise_summary(exact_summary)
    trajectory = make_denoising_trajectory(steps)

    # Preserve the old summary filename as a by-k compatibility alias while
    # also writing explicit schema-v2 filenames.
    size_summary.to_csv(output_dir / "summary_statistics.csv", index=False)
    size_summary.to_csv(
        output_dir / "summary_statistics_by_subset_size.csv", index=False
    )
    exact_summary.to_csv(
        output_dir / "summary_statistics_by_exact_subset.csv", index=False
    )
    if not pairwise_summary.empty:
        pairwise_summary.to_csv(
            output_dir / "pairwise_summary_statistics.csv", index=False
        )

    trajectory.to_csv(output_dir / "denoising_trajectory.csv", index=False)
    write_text_trajectory(
        trajectory,
        output_dir / "denoising_trajectory.txt",
    )

    print("Summary statistics by subset size")
    print("---------------------------------")
    print(compact_size_summary(size_summary).to_string(index=False))
    print()

    plot_histograms_by_size(
        analysis_estimates,
        output_path=output_dir / "factorization_gap_histograms.png",
        bins=args.bins,
        dpi=args.dpi,
        show=args.show,
    )
    plot_all_boxplots(
        analysis_estimates,
        output_path=output_dir / "factorization_gap_boxplots.png",
        dpi=args.dpi,
        show=args.show,
    )
    plot_size_trajectory(
        analysis_estimates,
        column="absolute_log_gap",
        ylabel="|log(P_S / Q_S)|",
        title="Absolute log gap over the denoising trajectory",
        output_path=output_dir / "absolute_log_gap_trajectory.png",
        dpi=args.dpi,
        show=args.show,
    )
    plot_ecdf_by_size(
        analysis_estimates,
        column="absolute_log_gap",
        xlabel="|log(P_S / Q_S)|",
        title="Absolute log-gap ECDF by subset size k",
        output_path=output_dir / "absolute_log_gap_ecdf.png",
        dpi=args.dpi,
        show=args.show,
    )
    plot_top_exact_subsets(
        exact_summary,
        top_n=args.top_subsets,
        output_path=(
            output_dir / "top_exact_subsets_by_median_absolute_log_gap.png"
        ),
        dpi=args.dpi,
        show=args.show,
    )

    write_report(
        size_summary=size_summary,
        exact_summary=exact_summary,
        pairwise_summary=pairwise_summary,
        trajectory=trajectory,
        output_dir=output_dir,
    )

    print()
    print(f"Run directory:   {run_dir}")
    print(f"Input estimate rows: {len(estimates)}")
    print(f"Analyzed rows:       {len(analysis_estimates)} (singletons included)")
    print(f"Rollout steps:   {len(steps)}")
    print(
        "Analyzed subsets:   "
        f"{analysis_estimates['subset_mask'].nunique()} (all available sizes)"
    )
    print(f"Output directory: {output_dir}")
    return 0


# Convert main's integer result into the shell process exit status.
if __name__ == "__main__":
    raise SystemExit(main())
