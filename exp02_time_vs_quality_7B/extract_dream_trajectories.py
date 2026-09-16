#!/usr/bin/env python3
"""Extract readable fixed and adaptive denoising trajectories for examples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from audit_dream_topk import load_json, read_jsonl, resolve_path
from run_dream_topk import atomic_write_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-run", type=Path, required=True)
    parser.add_argument("--adaptive-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--example-ids", type=int, nargs="*")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--default-example-count",
        type=int,
        default=3,
        help="Used only when --example-ids is omitted.",
    )
    args = parser.parse_args(argv)
    if args.default_example_count < 1:
        parser.error("--default-example-count must be positive.")
    if args.example_ids is not None:
        if any(value < 0 for value in args.example_ids):
            parser.error("Example IDs cannot be negative.")
        if len(set(args.example_ids)) != len(args.example_ids):
            parser.error("Example IDs must not contain duplicates.")
    return args


def escape_markdown(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def render_canvas(token_text: Sequence[str | None]) -> str:
    return escape_markdown(
        "".join("▢" if value is None else value for value in token_text)
    )


def validate_run_pair(
    fixed_run: Path,
    adaptive_run: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = load_json(fixed_run / "metadata.json")
    adaptive = load_json(adaptive_run / "metadata.json")
    fixed_audit = load_json(fixed_run / "audit.json")
    adaptive_audit = load_json(adaptive_run / "audit.json")
    if fixed.get("status") != "complete" or adaptive.get("status") != "complete":
        raise ValueError("Both experiment runs must be complete.")
    if (
        fixed_audit.get("status") != "verified"
        or adaptive_audit.get("status") != "verified"
    ):
        raise ValueError("Both experiment audits must be verified.")
    if fixed["dataset"]["sha256"] != adaptive["dataset"]["sha256"]:
        raise ValueError("Fixed and adaptive runs used different datasets.")
    for field in ("prefix_length", "continuation_length"):
        if fixed["experiment"][field] != adaptive["experiment"][field]:
            raise ValueError(f"Fixed and adaptive {field} values differ.")
    return fixed, adaptive


def choose_example_ids(
    args: argparse.Namespace,
    fixed: dict[str, Any],
    adaptive: dict[str, Any],
) -> list[int]:
    fixed_ids = {
        int(value)
        for value in fixed["experiment"]["trajectory_example_ids"]
    }
    adaptive_ids = {
        int(value)
        for value in adaptive["experiment"]["trajectory_example_ids"]
    }
    available = sorted(fixed_ids & adaptive_ids)
    if args.example_ids is None:
        selected = available[: args.default_example_count]
    else:
        selected = list(args.example_ids)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(
            "Requested examples were not recorded by both runs: "
            f"{missing}. Available IDs: {available}"
        )
    if not selected:
        raise ValueError("No trajectory examples were selected.")
    return selected


def add_fixed_steps(
    output: list[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    example_ids: Sequence[int],
    continuation_length: int,
) -> None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["k"]), int(row["example_id"]))
        if key[1] in example_ids:
            grouped.setdefault(key, []).append(row)
    k_values = sorted({key[0] for key in grouped})
    expected = {
        (k, example_id) for k in k_values for example_id in example_ids
    }
    if set(grouped) != expected:
        raise ValueError("Fixed trajectory coverage is incomplete.")

    for key in sorted(grouped):
        k, example_id = key
        token_ids: list[int | None] = [None] * continuation_length
        token_text: list[str | None] = [None] * continuation_length
        for row in sorted(grouped[key], key=lambda value: int(value["step_index"])):
            positions = [
                int(value) for value in row["selected_canvas_positions"]
            ]
            committed_ids = [
                int(value) for value in row["selected_token_ids"]
            ]
            committed_text = [
                str(value) for value in row["selected_token_text"]
            ]
            confidence = [
                float(value) for value in row["selected_confidences"]
            ]
            for position, token_id, text in zip(
                positions,
                committed_ids,
                committed_text,
            ):
                token_ids[position] = token_id
                token_text[position] = text
            output.append(
                {
                    "schedule_type": "fixed",
                    "schedule_label": f"k={k}",
                    "schedule_value": k,
                    "example_id": example_id,
                    "step_index": int(row["step_index"]),
                    "remaining_masks_before": int(
                        row["remaining_masks_before"]
                    ),
                    "candidate_canvas_positions": positions,
                    "candidate_token_ids": committed_ids,
                    "candidate_token_text": committed_text,
                    "candidate_confidences": confidence,
                    "accepted_prefix_length": len(positions),
                    "committed_canvas_positions": positions,
                    "committed_token_ids": committed_ids,
                    "committed_token_text": committed_text,
                    "remaining_masks_after": int(row["remaining_masks_after"]),
                    "canvas_token_ids_after": list(token_ids),
                    "canvas_token_text_after": list(token_text),
                }
            )


def add_adaptive_steps(
    output: list[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    example_ids: Sequence[int],
    continuation_length: int,
) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["threshold_label"]), int(row["example_id"]))
        if key[1] in example_ids:
            grouped.setdefault(key, []).append(row)
    labels = sorted({key[0] for key in grouped})
    expected = {
        (label, example_id)
        for label in labels
        for example_id in example_ids
    }
    if set(grouped) != expected:
        raise ValueError("Adaptive trajectory coverage is incomplete.")

    for key in sorted(grouped):
        label, example_id = key
        token_ids: list[int | None] = [None] * continuation_length
        token_text: list[str | None] = [None] * continuation_length
        for row in sorted(grouped[key], key=lambda value: int(value["step_index"])):
            candidate_positions = [
                int(value) for value in row["candidate_canvas_positions"]
            ]
            candidate_ids = [
                int(value) for value in row["candidate_token_ids"]
            ]
            candidate_text = [
                str(value) for value in row["candidate_token_text"]
            ]
            accepted = int(row["accepted_prefix_length"])
            positions = [
                int(value) for value in row["committed_canvas_positions"]
            ]
            committed_ids = [
                int(value) for value in row["committed_token_ids"]
            ]
            committed_text = candidate_text[:accepted]
            for position, token_id, text in zip(
                positions,
                committed_ids,
                committed_text,
            ):
                token_ids[position] = token_id
                token_text[position] = text
            output.append(
                {
                    "schedule_type": "adaptive",
                    "schedule_label": (
                        f"tau=log({float(row['tau_multiplier']):g})"
                    ),
                    "schedule_value": label,
                    "tau": float(row["tau"]),
                    "tau_multiplier": float(row["tau_multiplier"]),
                    "example_id": example_id,
                    "step_index": int(row["step_index"]),
                    "remaining_masks_before": int(
                        row["remaining_masks_before"]
                    ),
                    "candidate_canvas_positions": candidate_positions,
                    "candidate_token_ids": candidate_ids,
                    "candidate_token_text": candidate_text,
                    "candidate_confidences": row["candidate_confidences"],
                    "marginal_log_factors": row["marginal_log_factors"],
                    "conditional_log_factors": row[
                        "conditional_log_factors"
                    ],
                    "cumulative_log_marginals": row[
                        "cumulative_log_marginals"
                    ],
                    "cumulative_log_joints": row[
                        "cumulative_log_joints"
                    ],
                    "cumulative_errors": row["cumulative_errors"],
                    "conditional_probes": int(row["conditional_probes"]),
                    "accepted_prefix_length": accepted,
                    "committed_canvas_positions": positions,
                    "committed_token_ids": committed_ids,
                    "committed_token_text": committed_text,
                    "remaining_masks_after": int(row["remaining_masks_after"]),
                    "canvas_token_ids_after": list(token_ids),
                    "canvas_token_text_after": list(token_text),
                }
            )


def write_report(
    path: Path,
    records: Sequence[dict[str, Any]],
    dataset_by_id: dict[int, dict[str, Any]],
    example_ids: Sequence[int],
) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(
            (int(row["example_id"]), str(row["schedule_label"])),
            [],
        ).append(row)

    lines = [
        "# Dream denoising trajectories",
        "",
        "`▢` marks a continuation position that is still masked.",
        "",
    ]
    for example_id in example_ids:
        dataset_row = dataset_by_id[example_id]
        lines.extend(
            [
                f"## Example {example_id}",
                "",
                f"**Prefix:** {escape_markdown(dataset_row['prefix_text'])}",
                "",
                "**Human continuation:** "
                + escape_markdown(dataset_row["human_continuation_text"]),
                "",
            ]
        )
        schedule_keys = sorted(
            key for key in grouped if key[0] == example_id
        )
        for key in schedule_keys:
            schedule_rows = sorted(
                grouped[key],
                key=lambda row: int(row["step_index"]),
            )
            lines.extend(
                [
                    f"### {key[1]}",
                    "",
                    "| Step | Remaining | Accepted | New commits | Canvas after step |",
                    "|---:|---:|---:|---|---|",
                ]
            )
            for row in schedule_rows:
                commits = ", ".join(
                    f"{position}:{escape_markdown(text)}"
                    for position, text in zip(
                        row["committed_canvas_positions"],
                        row["committed_token_text"],
                    )
                )
                lines.append(
                    f"| {row['step_index']} "
                    f"| {row['remaining_masks_before']}→"
                    f"{row['remaining_masks_after']} "
                    f"| {row['accepted_prefix_length']} "
                    f"| {commits} "
                    f"| {render_canvas(row['canvas_token_text_after'])} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plain_token(value: str) -> str:
    """Make whitespace and control characters visible in a text report."""

    return json.dumps(str(value), ensure_ascii=False)


def plain_canvas(token_text: Sequence[str | None]) -> str:
    return "".join("▢" if value is None else value for value in token_text)


def format_plain_example(
    example_id: int,
    records: Sequence[dict[str, Any]],
    dataset_row: dict[str, Any],
) -> str:
    lines = [
        "=" * 100,
        f"EXAMPLE {example_id}",
        "=" * 100,
        f"PREFIX: {dataset_row['prefix_text']}",
        f"HUMAN CONTINUATION: {dataset_row['human_continuation_text']}",
        "",
        (
            "Adaptive error is the cumulative log-space magnitude "
            "|log J_m - log M_m|. exp(error) is the corresponding "
            "multiplicative discrepancy."
        ),
        "",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        if int(row["example_id"]) == example_id:
            grouped.setdefault(
                (
                    str(row["schedule_type"]),
                    str(row["schedule_label"]),
                ),
                [],
            ).append(row)

    def schedule_order(key: tuple[str, str]) -> tuple[int, float, str]:
        schedule_type, label = key
        group = grouped[key]
        if schedule_type == "fixed":
            value = float(group[0]["schedule_value"])
            return (0, value, label)
        return (1, float(group[0]["tau_multiplier"]), label)

    for key in sorted(grouped, key=schedule_order):
        schedule_type, label = key
        schedule_rows = sorted(
            grouped[key],
            key=lambda row: int(row["step_index"]),
        )
        lines.extend(
            [
                "-" * 100,
                f"SCHEDULE: {label} ({schedule_type})",
                "-" * 100,
            ]
        )
        for row in schedule_rows:
            lines.append(
                f"STEP {int(row['step_index']):03d} | masks "
                f"{row['remaining_masks_before']} -> "
                f"{row['remaining_masks_after']} | accepted "
                f"{row['accepted_prefix_length']}"
            )
            positions = row["candidate_canvas_positions"]
            token_ids = row["candidate_token_ids"]
            token_text = row["candidate_token_text"]
            confidence = row["candidate_confidences"]
            if schedule_type == "fixed":
                lines.append(
                    "  selected rank | canvas position | token id | "
                    "token text | confidence"
                )
                for rank, values in enumerate(
                    zip(positions, token_ids, token_text, confidence),
                    start=1,
                ):
                    position, token_id, text, probability = values
                    lines.append(
                        f"  {rank:>13} | {int(position):>15} | "
                        f"{int(token_id):>8} | {plain_token(text):<18} | "
                        f"{float(probability):.9g}"
                    )
            else:
                errors = [
                    float(value) for value in row["cumulative_errors"]
                ]
                log_marginals = [
                    float(value)
                    for value in row["cumulative_log_marginals"]
                ]
                log_joints = [
                    float(value) for value in row["cumulative_log_joints"]
                ]
                accepted = int(row["accepted_prefix_length"])
                lines.append(
                    "  frozen rank | canvas position | token id | token text "
                    "| confidence | decision | log M_m | log J_m | "
                    "abs log error | exp(error)"
                )
                for index, values in enumerate(
                    zip(positions, token_ids, token_text, confidence),
                ):
                    position, token_id, text, probability = values
                    rank = index + 1
                    if index < len(errors):
                        error = errors[index]
                        log_m = log_marginals[index]
                        log_j = log_joints[index]
                        gap = math.exp(error) if error < 709.0 else math.inf
                        numeric = (
                            f"{log_m:>9.5f} | {log_j:>9.5f} | "
                            f"{error:>13.6g} | {gap:.6g}"
                        )
                    else:
                        numeric = (
                            "not probed | not probed | not probed    | "
                            "not probed"
                        )
                    if rank <= accepted:
                        decision = "COMMITTED"
                    elif index < len(errors):
                        decision = "FIRST FAILURE"
                    else:
                        decision = "NOT PROBED"
                    lines.append(
                        f"  {rank:>11} | {int(position):>15} | "
                        f"{int(token_id):>8} | {plain_token(text):<18} | "
                        f"{float(probability):.9g} | {decision:<13} | "
                        f"{numeric}"
                    )
                lines.append(
                    f"  conditional probes executed: "
                    f"{row['conditional_probes']}"
                )
            lines.append(
                "  canvas after step: "
                + plain_canvas(row["canvas_token_text_after"])
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_plain_reports(
    output_dir: Path,
    records: Sequence[dict[str, Any]],
    dataset_by_id: dict[int, dict[str, Any]],
    example_ids: Sequence[int],
) -> tuple[Path, list[Path]]:
    reports: list[tuple[int, str]] = []
    paths: list[Path] = []
    for example_id in example_ids:
        text = format_plain_example(
            example_id,
            records,
            dataset_by_id[example_id],
        )
        path = output_dir / f"trajectory_example_{example_id}.txt"
        path.write_text(text, encoding="utf-8")
        reports.append((example_id, text))
        paths.append(path)
    combined = output_dir / "trajectory_all.txt"
    combined.write_text(
        "\n\n".join(text for _, text in reports),
        encoding="utf-8",
    )
    return combined, paths


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fixed_run = args.fixed_run.expanduser().resolve()
    adaptive_run = args.adaptive_run.expanduser().resolve()
    fixed, adaptive = validate_run_pair(fixed_run, adaptive_run)
    selected_ids = choose_example_ids(args, fixed, adaptive)
    dataset = (
        args.dataset.expanduser().resolve()
        if args.dataset
        else Path(fixed["dataset"]["path"]).expanduser().resolve()
    )
    dataset_rows = read_jsonl(dataset)
    dataset_by_id = {
        int(row["example_id"]): row for row in dataset_rows
    }
    missing = [value for value in selected_ids if value not in dataset_by_id]
    if missing:
        raise ValueError(f"Examples are absent from the dataset: {missing}")

    continuation_length = int(fixed["experiment"]["continuation_length"])
    extracted: list[dict[str, Any]] = []
    add_fixed_steps(
        extracted,
        read_jsonl(
            resolve_path(
                fixed["artifacts"]["trajectories"],
                base=fixed_run,
            )
        ),
        example_ids=selected_ids,
        continuation_length=continuation_length,
    )
    add_adaptive_steps(
        extracted,
        read_jsonl(
            resolve_path(
                adaptive["artifacts"]["trajectories"],
                base=adaptive_run,
            )
        ),
        example_ids=selected_ids,
        continuation_length=continuation_length,
    )
    extracted.sort(
        key=lambda row: (
            int(row["example_id"]),
            str(row["schedule_type"]),
            str(row["schedule_label"]),
            int(row["step_index"]),
        )
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "trajectory_steps.jsonl"
    report_path = output_dir / "trajectory_report.md"
    atomic_write_jsonl(jsonl_path, extracted)
    write_report(
        report_path,
        extracted,
        dataset_by_id,
        selected_ids,
    )
    text_path, example_text_paths = write_plain_reports(
        output_dir,
        extracted,
        dataset_by_id,
        selected_ids,
    )
    manifest = {
        "dataset_sha256": fixed["dataset"]["sha256"],
        "example_ids": selected_ids,
        "prefix_length": fixed["experiment"]["prefix_length"],
        "continuation_length": continuation_length,
        "fixed_run": str(fixed_run),
        "adaptive_run": str(adaptive_run),
        "trajectory_steps": str(jsonl_path),
        "report": str(report_path),
        "text_report": str(text_path),
        "per_example_text_reports": [
            str(path) for path in example_text_paths
        ],
    }
    (output_dir / "trajectory_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] trajectory report: {report_path}")
    print(f"[done] plain-text report: {text_path}")
    print(f"[done] trajectory data:   {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
