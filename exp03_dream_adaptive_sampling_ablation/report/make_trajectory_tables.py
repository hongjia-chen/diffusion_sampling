#!/usr/bin/env python3
"""Build per-step adaptive trajectory tables and machine-readable CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
BENCHMARKS = ("gsm8k", "mbpp_plus", "schema_json")
BENCHMARK_LABELS = {
    "gsm8k": "GSM8K",
    "mbpp_plus": "MBPP+",
    "schema_json": "Schema JSON",
}
DELTAS = ("0p10", "0p15", "0p20", "0p30", "0p50", "0p75", "1p00", "1p50")
METHODS = tuple("adaptive_delta_" + value for value in DELTAS)
ROW_END = r" \\"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate(method_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Return one row per 1-based step; means are conditional on reaching it."""
    if not method_rows:
        return []
    maximum = max(
        len(row["commitments_by_step"])
        for rows in method_rows.values()
        for row in rows
    )
    output = []
    for step in range(1, maximum + 1):
        result: dict[str, Any] = {"step": step}
        for method, rows in method_rows.items():
            values = [
                int(row["commitments_by_step"][step - 1])
                for row in rows
                if len(row["commitments_by_step"]) >= step
            ]
            result[f"{method}_mean_tokens"] = (
                sum(values) / len(values) if values else math.nan
            )
            result[f"{method}_reached"] = len(values)
        output.append(result)
    return output


def load_benchmark(output_root: Path, benchmark: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    reference_ids: set[str] | None = None
    for method in METHODS:
        path = output_root / benchmark / method / "trajectories.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run scripts/replay_adaptive_trajectories.py first"
            )
        rows = read_jsonl(path)
        ids = [str(row["example_id"]) for row in rows]
        if len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError(
                f"{path} must contain exactly 100 unique tasks; found {len(set(ids))}"
            )
        if any(row.get("method") != method for row in rows):
            raise ValueError(f"method mismatch in {path}")
        for row in rows:
            widths = row.get("commitments_by_step")
            if not isinstance(widths, list) or len(widths) != int(row["denoising_steps"]):
                raise ValueError(f"invalid trajectory for {method}/{row.get('example_id')}")
        current_ids = set(ids)
        if reference_ids is None:
            reference_ids = current_ids
        elif current_ids != reference_ids:
            raise ValueError(f"methods are not paired on identical tasks for {benchmark}")
        result[method] = rows
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step"] + [
        f"{method}_{suffix}"
        for method in METHODS
        for suffix in ("mean_tokens", "reached")
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def delta_label(method: str) -> str:
    return method.removeprefix("adaptive_delta_").replace("p", ".")


def latex_table(benchmark: str, rows: list[dict[str, Any]]) -> str:
    groups = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{${delta_label(method)}$}}" for method in METHODS
    )
    subheaders = " & ".join("Mean & $n$" for _ in METHODS)
    body = []
    for row in rows:
        cells = [str(row["step"])]
        for method in METHODS:
            reached = int(row[f"{method}_reached"])
            mean = row[f"{method}_mean_tokens"]
            cells.extend(["---" if reached == 0 else f"{mean:.2f}", str(reached)])
        body.append(" & ".join(cells) + ROW_END)
    caption = (
        f"Adaptive generation trajectory on {BENCHMARK_LABELS[benchmark]}. "
        "For each $\\delta$, Mean is the mean tokens committed at that step among "
        "tasks that reached it, and $n$ is the number of the 100 tasks that reached "
        "it. Step numbers are one-based."
    )
    return "\n".join(
        [
            r"\begin{landscape}",
            r"\begingroup\tiny",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\renewcommand{\arraystretch}{0.92}",
            r"\setlength{\LTcapwidth}{\linewidth}",
            r"\begin{longtable}{r*{8}{rr}}",
            f"\\caption{{{caption}}}\\label{{tab:trajectory-{benchmark.replace('_', '-')}}}{ROW_END}",
            r"\toprule",
            f"Step & {groups}{ROW_END}",
            f" & {subheaders}{ROW_END}",
            r"\midrule",
            r"\endfirsthead",
            r"\multicolumn{17}{l}{\itshape Table \thetable\ (continued)}\\",
            r"\toprule",
            f"Step & {groups}{ROW_END}",
            f" & {subheaders}{ROW_END}",
            r"\midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endlastfoot",
            *body,
            r"\end{longtable}",
            r"\endgroup",
            r"\end{landscape}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "v3_matched")
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    args = parser.parse_args()
    for benchmark in BENCHMARKS:
        rows = aggregate(load_benchmark(args.output_root, benchmark))
        csv_path = args.report_dir / "derived" / f"trajectory_{benchmark}.csv"
        tex_path = args.report_dir / f"table_trajectory_{benchmark}.tex"
        write_csv(csv_path, rows)
        tex_path.write_text(latex_table(benchmark, rows), encoding="utf-8")
        print(f"wrote {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
