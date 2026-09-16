#!/usr/bin/env python3
"""Quality against throughput, one full-width figure per benchmark.

X is output tokens per second (log), Y is the benchmark's primary score. Every
scheduler on one axis; no second y-scale anywhere. The fixed:k sweeps are drawn
as lines because k is an ordered knob with one point per value; the adaptive
delta points are left unconnected, as asked, and each is labelled with its delta
instead so the sweep order is still readable.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    BENCH_LABEL,
    BENCHMARKS,
    COLOR,
    DELTA_VALUE,
    GROUPS,
    MARKER,
    METRIC_LABEL,
    REPORT,
    summary_rows,
)

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
SURFACE = "#fcfcfb"

LINESTYLE = {
    "1_dream7b_baseline": "-",
    "2_sdar_b64_baseline": "--",
    "3_adaptive_delta": "none",
    "4_autospec": "none",
}

# Per-point label offsets (points). The adaptive cluster is tight on MATH-500
# and HumanEval+, so its five labels fan out left/right rather than stacking.
FIXED_OFFSET = {"1_dream7b_baseline": (0, 9), "2_sdar_b64_baseline": (0, -15)}
DELTA_OFFSET = {
    "adaptive_delta_0p10": (-16, 7),
    "adaptive_delta_0p20": (0, 10),
    "adaptive_delta_0p50": (16, 7),
    "adaptive_delta_1p00": (-16, -14),
    "adaptive_delta_1p50": (16, -14),
}


def label_for(row) -> tuple[str, tuple[int, int]] | None:
    method, group = row["method"], row["group"]
    if method in ("fixed_k_1", "fixed_k_8"):
        # Ends of the fixed sweeps only: the line already carries the order, and
        # SDAR k=2 sits inside the adaptive cluster where a label would collide.
        return f"k={method.rsplit('_', 1)[1]}", FIXED_OFFSET[group]
    if method.startswith("adaptive_delta"):
        return rf"$\delta$={DELTA_VALUE[method]:g}", DELTA_OFFSET[method]
    return None


def draw(bench: str, rows) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for group in GROUPS:
        got = sorted(
            (r for r in rows if r["group"] == group and r["benchmark"] == bench),
            key=lambda r: r["tokens_per_sec"],
        )
        if not got:
            continue
        xs = [r["tokens_per_sec"] for r in got]
        ys = [r["score"] for r in got]
        ax.plot(
            xs,
            ys,
            linestyle=LINESTYLE[group],
            marker=MARKER[group],
            color=COLOR[group],
            markersize=8.5,
            linewidth=2.0,
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            label=GROUPS[group],
            zorder=3,
        )
        for r, x, y in zip(got, xs, ys):
            got_label = label_for(r)
            if got_label is None:
                continue
            text, offset = got_label
            ax.annotate(
                text, (x, y), textcoords="offset points", xytext=offset,
                ha="center", va="center", fontsize=8, color=INK_2, zorder=4,
            )
        if group == "4_autospec":
            ax.annotate(
                "autospec", (xs[0], ys[0]), textcoords="offset points",
                xytext=(10, 0), ha="left", va="center", fontsize=8.5,
                color=COLOR[group], zorder=4,
            )

    ax.set_xscale("log")
    ax.set_title(BENCH_LABEL[bench], fontsize=13, color=INK, loc="left", pad=10)
    ax.set_xlabel("output tokens / second (log scale)", fontsize=10, color=INK_2)
    ax.set_ylabel(METRIC_LABEL[bench], fontsize=10, color=INK_2)
    ax.set_ylim(-0.08, 0.92)
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color="#f1f0ec", linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=9, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d8d7d1")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
        labelcolor=INK_2, handlelength=2.4, bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = REPORT / "figures" / f"frontier_{bench}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    rows = summary_rows()
    for bench in BENCHMARKS:
        draw(bench, rows)


if __name__ == "__main__":
    main()
