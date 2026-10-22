#!/usr/bin/env python3
"""The η study's measurement run (ETA_STUDY.md): one dataset, one output directory.

SDAR-8B decodes every prompt greedily at fixed:1, exactly as the B=16 baseline
does. At one random step per block, the probe rows of that step (row j fills the
j-1 most confident candidates) go through the 8B and the 1.7B, and
`sdar_harness.dependency` records, for every rank j, D^P, D^V, η and the
diagnostics listed in ETA_STUDY.md. The probes never touch the decode.

    python measure_eta.py --dataset ../data/heldout/gsm8k_train.jsonl \
        --output-dir outputs/eta/gsm8k_train \
        --proposer ../models/SDAR-8B-Chat-b16-local --proposer-devices cuda:0 \
        --reference ../models/SDAR-1.7B-Chat-b16-local --reference-device cuda:1

Writes `steps.jsonl` (one line per measured step), `tasks.jsonl` (one line per
decoded prompt) and `run_metadata.json`. A task's steps are appended before its
task line, so a rerun resumes after the last complete task and drops the
partial one's steps. `--check-against` names a fixed:1 baseline's records.jsonl;
each task then records whether its completion is byte-identical to the baseline's.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent / "harness"
sys.path.insert(0, str(HARNESS))

import torch  # noqa: E402

from sdar_harness.dependency import MeasureConfig, generate_measured  # noqa: E402
from sdar_harness.io import sha256_file, stream_jsonl, write_json, write_jsonl  # noqa: E402
from sdar_harness.model import (  # noqa: E402
    MaskedDiffusionProvider,
    ModelOptions,
    assert_shared_vocabulary,
)


def load(path: str, devices: list[str], args: argparse.Namespace) -> MaskedDiffusionProvider:
    return MaskedDiffusionProvider.from_pretrained(ModelOptions(
        model_id=path, family="sdar", block_length=args.block_length,
        device=devices[0], devices=tuple(devices) if len(devices) > 1 else None,
        dtype=args.dtype, local_files_only=args.local_files_only,
        require_lean_logits=True, vocab_limit=args.vocab_limit,
    ))


def completed_tasks(output_dir: Path) -> set[str]:
    """IDs with a task line; steps of any other task (a crash mid-task) are dropped."""
    tasks_path, steps_path = output_dir / "tasks.jsonl", output_dir / "steps.jsonl"
    done = {str(row["example_id"]) for row in stream_jsonl(tasks_path)} if tasks_path.exists() else set()
    if steps_path.exists():
        steps = list(stream_jsonl(steps_path))
        kept = [row for row in steps if str(row["example_id"]) in done]
        if len(kept) != len(steps):
            print(f"resume: dropping {len(steps) - len(kept)} steps of an unfinished task", flush=True)
            write_jsonl(steps_path, kept)
    return done


def append(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_state() -> dict:
    def run(*command: str) -> str:
        try:
            return subprocess.run(command, cwd=HARNESS, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty_files": len(run("git", "status", "--porcelain").splitlines()),
            "dependency_py_sha256": sha256_file(HARNESS / "sdar_harness/dependency.py")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proposer", required=True, help="SDAR-8B checkpoint (decodes and is P)")
    parser.add_argument("--proposer-devices", nargs="+", default=["cuda:0"],
                        help="one card, or several to pipeline-split the 8B")
    parser.add_argument("--reference", required=True, help="SDAR-1.7B checkpoint (V)")
    parser.add_argument("--reference-device", default="cuda:1")
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--continuation-length", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--vocab-limit", type=int, default=151936)
    parser.add_argument("--group-rows", type=int, default=16,
                        help="probe rows per model call (16 = a whole B=16 step in one call)")
    parser.add_argument("--measure-per-block", type=int, default=1)
    parser.add_argument("--max-measured", type=int, default=64, help="measured steps per task, at most")
    parser.add_argument("--floor-tasks", type=int, default=10,
                        help="the first N tasks also read the 8B's rows under --floor-group-rows")
    parser.add_argument("--floor-group-rows", type=int, default=4)
    parser.add_argument("--limit", type=int, help="only the first N tasks of the dataset")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--check-against", type=Path, help="a fixed:1 baseline's records.jsonl")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    rows = list(stream_jsonl(args.dataset))
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = completed_tasks(args.output_dir)
    baseline = None
    if args.check_against is not None:
        baseline = {str(r["example_id"]): r["raw_completion"] for r in stream_jsonl(args.check_against)}

    torch.manual_seed(args.seed)
    load_began = time.perf_counter()
    proposer = load(args.proposer, args.proposer_devices, args)
    reference = load(args.reference, [args.reference_device], args)
    load_seconds = time.perf_counter() - load_began
    assert_shared_vocabulary(proposer.tokenizer, reference.tokenizer, args.vocab_limit)
    if proposer.mask_token_id != reference.mask_token_id or proposer.stop_token_ids != reference.stop_token_ids:
        raise ValueError("the two checkpoints disagree on the mask or stop tokens")

    devices = list(proposer.devices) + list(reference.devices)
    metadata = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256_file(args.dataset),
        "number_of_tasks": len(rows),
        "proposer_devices": [str(d) for d in proposer.devices], "proposer_layer_split": proposer.layer_split,
        "reference_devices": [str(d) for d in reference.devices],
        "gpu_names": sorted({torch.cuda.get_device_name(d) for d in devices if torch.device(d).type == "cuda"}),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "load_seconds": load_seconds,
        "lean_path_relative_gap": {"proposer": getattr(proposer, "lean_path_relative_gap", None),
                                   "reference": getattr(reference, "lean_path_relative_gap", None)},
        "mask_token_id": proposer.mask_token_id, "stop_token_ids": sorted(proposer.stop_token_ids),
        "excluded_token_ids": sorted(proposer.excluded_token_ids),
        "python": sys.version.split()[0], "torch": torch.__version__, "host": platform.node(),
        "code": git_state(),
    }
    write_json(args.output_dir / f"run_metadata_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json", metadata)

    main_config = MeasureConfig(
        continuation_length=args.continuation_length, group_rows=args.group_rows,
        measure_per_block=args.measure_per_block, max_measured=args.max_measured,
    )
    floor_config = MeasureConfig(
        continuation_length=args.continuation_length, group_rows=args.group_rows,
        measure_per_block=args.measure_per_block, max_measured=args.max_measured,
        floor_group_rows=args.floor_group_rows,
    )
    started = time.perf_counter()
    for task_index, row in enumerate(rows):
        example_id = str(row["example_id"])
        if example_id in done:
            continue
        rendered = proposer.render_prompt(row["prompt"])
        prompt_ids = proposer.encode_prompt(rendered)
        if not torch.equal(prompt_ids.cpu(), reference.encode_prompt(reference.render_prompt(row["prompt"])).cpu()):
            raise ValueError(f"{example_id}: the two tokenizers encode the prompt differently")
        limit = getattr(proposer.model.config, "max_position_embeddings", None)
        if limit is not None and int(prompt_ids.shape[1]) + args.continuation_length > int(limit):
            raise ValueError(f"{example_id}: prompt + canvas exceeds the model's {limit} positions")
        floor = task_index < args.floor_tasks
        for device in devices:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
        began = time.perf_counter()
        result = generate_measured(
            proposer, reference, prompt_ids, floor_config if floor else main_config,
            rng=random.Random(f"{args.seed}:{example_id}"),
        )
        for device in devices:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
        wall = time.perf_counter() - began
        completion = proposer.decode(result.continuation_token_ids)
        benchmark = row.get("benchmark", args.dataset.stem)
        steps = [{"example_id": example_id, "benchmark": benchmark, **record} for record in result.records]
        task = {
            "example_id": example_id, "benchmark": benchmark, "task_index": task_index,
            "prompt_tokens": int(prompt_ids.shape[1]),
            "output_tokens": len(result.continuation_token_ids),
            "raw_completion": completion,
            "stop_token_id": result.stop_token_id, "eos_position": result.eos_position,
            "maximum_length_termination": result.eos_position is None,
            "early_stopped": result.early_stopped,
            "denoising_steps": result.denoising_steps, "commit_passes": result.commit_passes,
            "measured_steps": len(result.records), "floor": floor,
            "probe_forwards": result.probe_forwards,
            "wall_seconds": round(wall, 3), "measure_seconds": round(result.measure_seconds, 3),
            "peak_gpu_memory_bytes": {
                str(d): torch.cuda.max_memory_allocated(d) for d in devices if torch.device(d).type == "cuda"
            },
        }
        if baseline is not None:
            task["matches_baseline"] = baseline.get(example_id) == completion if example_id in baseline else None
        append(args.output_dir / "steps.jsonl", steps)
        append(args.output_dir / "tasks.jsonl", [task])
        elapsed = time.perf_counter() - started
        match = "" if baseline is None else f" baseline={task['matches_baseline']}"
        print(f"[{task_index + 1}/{len(rows)}] {example_id}: {task['output_tokens']} tok, "
              f"{task['measured_steps']} measured, {wall:.1f}s ({result.measure_seconds:.1f}s probing)"
              f"{match}  elapsed {elapsed / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
