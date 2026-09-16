#!/usr/bin/env python3
"""Generate and time deterministic Dream top-k continuation schedules."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_MODEL = "Dream-org/Dream-v0-Base-7B"


@dataclass(frozen=True)
class PreparedBatch:
    example_ids: tuple[int, ...]
    state: Any


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-length", type=int, default=64)
    parser.add_argument("--continuation-length", type=int, default=128)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument(
        "--trajectory-example-ids",
        type=int,
        nargs="*",
        default=list(range(10)),
    )
    parser.add_argument(
        "--determinism-check-examples",
        type=int,
        default=8,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-special-tokens", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.prefix_length < 1:
        parser.error("--prefix-length must be at least 1.")
    if args.continuation_length < 1:
        parser.error("--continuation-length must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be at least 1.")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches cannot be negative.")
    if args.determinism_check_examples < 0:
        parser.error("--determinism-check-examples cannot be negative.")
    if not args.k_values or any(k < 1 for k in args.k_values):
        parser.error("--k-values must contain positive integers.")
    if len(set(args.k_values)) != len(args.k_values):
        parser.error("--k-values must not contain duplicates.")
    if len(set(args.trajectory_example_ids)) != len(
        args.trajectory_example_ids
    ):
        parser.error("--trajectory-example-ids must not contain duplicates.")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def default_dataset_metadata_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(".metadata.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def load_and_validate_dataset(
    dataset_path: Path,
    metadata_path: Path,
    *,
    prefix_length: int,
    continuation_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {metadata_path}")

    metadata = load_json(metadata_path)
    digest = sha256_file(dataset_path)
    if digest != metadata.get("dataset_sha256"):
        raise ValueError("Dataset checksum does not match its metadata.")
    if int(metadata.get("prefix_length", -1)) != prefix_length:
        raise ValueError("Requested prefix length disagrees with dataset metadata.")
    if int(metadata.get("continuation_length", -1)) != continuation_length:
        raise ValueError(
            "Requested continuation length disagrees with dataset metadata."
        )

    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_documents: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            example_id = int(record["example_id"])
            if example_id in seen_ids:
                raise ValueError(f"Duplicate example_id={example_id}.")
            seen_ids.add(example_id)
            document_id = str(record["document_id"])
            if document_id in seen_documents:
                raise ValueError(f"Duplicate document_id at line {line_number}.")
            seen_documents.add(document_id)
            if len(record["prefix_token_ids"]) != prefix_length:
                raise ValueError(f"Wrong prefix length at line {line_number}.")
            if (
                len(record["human_continuation_token_ids"])
                != continuation_length
            ):
                raise ValueError(f"Wrong continuation length at line {line_number}.")
            records.append(record)

    records.sort(key=lambda row: int(row["example_id"]))
    expected_ids = list(range(len(records)))
    if [int(row["example_id"]) for row in records] != expected_ids:
        raise ValueError("example_id values must be contiguous from zero.")
    if len(records) != int(metadata.get("num_examples", -1)):
        raise ValueError("Dataset record count disagrees with metadata.")
    return records, metadata, digest


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. "
                "Use --overwrite only for an intentional replacement."
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True)
    (path / "generations").mkdir()


def resolve_torch_dtype(torch: Any, name: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hardware_description(torch: Any, device: Any) -> str:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        return (
            f"{properties.name}; "
            f"{properties.total_memory / 1024**3:.1f} GiB"
        )
    return platform.processor() or platform.machine()


def prepare_batches(
    adapter: Any,
    torch: Any,
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    continuation_length: int,
) -> list[PreparedBatch]:
    batches: list[PreparedBatch] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        prefix_ids = torch.tensor(
            [row["prefix_token_ids"] for row in chunk],
            dtype=torch.long,
        )
        state = adapter.prepare_prefix_batch(
            prefix_ids,
            canvas_length=continuation_length,
        )
        batches.append(
            PreparedBatch(
                example_ids=tuple(int(row["example_id"]) for row in chunk),
                state=state,
            )
        )
    return batches


def warm_up(
    adapter: Any,
    torch: Any,
    device: Any,
    batches: Sequence[PreparedBatch],
    *,
    k: int,
    warmup_batches: int,
) -> None:
    for batch in batches[:warmup_batches]:
        adapter.generate_dream(batch.state, k=k)
    synchronize(torch, device)


def time_schedule(
    adapter: Any,
    torch: Any,
    device: Any,
    batches: Sequence[PreparedBatch],
    *,
    k: int,
    timing_repeat: int,
    retain_outputs: bool,
) -> tuple[dict[str, Any], Any | None]:
    device_outputs: list[Any] = []
    completed_samples = 0

    synchronize(torch, device)
    started = time.perf_counter()
    for batch in batches:
        result = adapter.generate_dream(batch.state, k=k)
        # Retain device references on every repetition so the Python-side path
        # is identical. Only the canonical repetition copies them to CPU.
        device_outputs.append(result.continuation_token_ids)
        completed_samples += len(batch.example_ids)
    synchronize(torch, device)
    elapsed = time.perf_counter() - started

    canonical = None
    if retain_outputs:
        canonical = torch.cat(
            [tensor.detach().cpu() for tensor in device_outputs],
            dim=0,
        )

    timing = {
        "k": k,
        "timing_repeat": timing_repeat,
        "completed_samples": completed_samples,
        "elapsed_seconds": elapsed,
        "amortized_seconds_per_sample": elapsed / completed_samples,
    }
    return timing, canonical


def decode_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        [int(token_id) for token_id in token_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def save_generations(
    path: Path,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    generated_ids: Any,
    *,
    k: int,
    continuation_length: int,
    mask_token_id: int,
) -> None:
    rows = generated_ids.tolist()
    if len(rows) != len(records):
        raise RuntimeError("Generated row count does not match the dataset.")

    output: list[dict[str, Any]] = []
    for dataset_record, token_ids in zip(records, rows):
        if len(token_ids) != continuation_length:
            raise RuntimeError("A generated continuation has the wrong length.")
        if mask_token_id in token_ids:
            raise RuntimeError("A generated continuation still contains MASK.")
        output.append(
            {
                "k": k,
                "example_id": int(dataset_record["example_id"]),
                "generated_token_ids": [int(token) for token in token_ids],
                "generated_text": decode_text(tokenizer, token_ids),
            }
        )
    atomic_write_jsonl(path, output)


def verify_determinism(
    adapter: Any,
    torch: Any,
    records: Sequence[dict[str, Any]],
    canonical_ids: Any,
    *,
    k: int,
    example_count: int,
    continuation_length: int,
    batch_size: int,
) -> bool:
    count = min(example_count, len(records))
    if count == 0:
        return True
    # Replay with exactly the same batch partition used by the timed pass.
    # Changing a BF16 matmul from B=1 to B>1 can perturb nearly tied logits;
    # greedy denoising then amplifies one changed argmax over later steps.
    repeated_chunks = []
    for start in range(0, count, batch_size):
        chunk = records[start : min(start + batch_size, count)]
        state = adapter.prepare_prefix_batch(
            torch.tensor(
                [row["prefix_token_ids"] for row in chunk],
                dtype=torch.long,
            ),
            canvas_length=continuation_length,
        )
        repeated_chunks.append(
            adapter.generate_dream(
                state,
                k=k,
            ).continuation_token_ids.detach().cpu()
        )
    repeated = torch.cat(repeated_chunks, dim=0)
    return bool(torch.equal(repeated, canonical_ids[:count]))


def collect_trajectories(
    adapter: Any,
    torch: Any,
    records: Sequence[dict[str, Any]],
    trajectory_ids: Sequence[int],
    *,
    k_values: Sequence[int],
    continuation_length: int,
) -> list[dict[str, Any]]:
    if not trajectory_ids:
        return []
    by_id = {int(record["example_id"]): record for record in records}
    missing = [example_id for example_id in trajectory_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Unknown trajectory example IDs: {missing}")
    selected_records = [by_id[example_id] for example_id in trajectory_ids]
    rows: list[dict[str, Any]] = []

    for k in k_values:
        state = adapter.prepare_prefix_batch(
            torch.tensor(
                [record["prefix_token_ids"] for record in selected_records],
                dtype=torch.long,
            ),
            canvas_length=continuation_length,
        )
        step_index = 0
        while step_index < math.ceil(continuation_length / k):
            step = adapter.dream_step(state, k=k)
            canvas_positions = step.selected_canvas_positions.detach().cpu().tolist()
            sequence_positions = (
                step.selected_sequence_positions.detach().cpu().tolist()
            )
            token_ids = step.selected_token_ids.detach().cpu().tolist()
            confidences = step.selected_confidences.detach().cpu().tolist()

            for batch_index, example_id in enumerate(trajectory_ids):
                selected_ids = [int(value) for value in token_ids[batch_index]]
                rows.append(
                    {
                        "k": k,
                        "example_id": int(example_id),
                        "step_index": step_index,
                        "remaining_masks_before": step.remaining_masks_before,
                        "selected_canvas_positions": [
                            int(value)
                            for value in canvas_positions[batch_index]
                        ],
                        "selected_sequence_positions": [
                            int(value)
                            for value in sequence_positions[batch_index]
                        ],
                        "selected_token_ids": selected_ids,
                        "selected_token_text": [
                            decode_text(adapter.tokenizer, [token_id])
                            for token_id in selected_ids
                        ],
                        "selected_confidences": [
                            float(value)
                            for value in confidences[batch_index]
                        ],
                        "remaining_masks_after": step.remaining_masks_after,
                    }
                )
            state = step.state
            step_index += 1
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run Dream.") from exc
    try:
        from dream_adapter import DreamAdapter, ModelLoadOptions
    except ImportError as exc:
        raise RuntimeError(
            "dream_adapter.py must be beside run_dream_topk.py."
        ) from exc

    dataset_path = args.dataset.expanduser().resolve()
    dataset_metadata_path = (
        args.dataset_metadata.expanduser().resolve()
        if args.dataset_metadata
        else default_dataset_metadata_path(dataset_path)
    )
    output_dir = args.output_dir.expanduser().resolve()
    records, dataset_metadata, dataset_digest = load_and_validate_dataset(
        dataset_path,
        dataset_metadata_path,
        prefix_length=args.prefix_length,
        continuation_length=args.continuation_length,
    )
    prepare_output_directory(output_dir, overwrite=args.overwrite)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device {device}.")
    dtype = resolve_torch_dtype(torch, args.dtype)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    metadata_path = output_dir / "metadata.json"
    timing_path = output_dir / "timings.jsonl"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": utc_now(),
        "dataset": {
            "path": str(dataset_path),
            "metadata_path": str(dataset_metadata_path),
            "sha256": dataset_digest,
            "num_examples": len(records),
        },
        "experiment": {
            "prefix_length": args.prefix_length,
            "continuation_length": args.continuation_length,
            "model_input_length": 1
            + args.prefix_length
            + args.continuation_length,
            "k_values": args.k_values,
            "batch_size": args.batch_size,
            "timing_repeats": args.timing_repeats,
            "warmup_batches": args.warmup_batches,
            "trajectory_example_ids": args.trajectory_example_ids,
            "determinism_check_examples": args.determinism_check_examples,
            "seed": args.seed,
        },
        "model": {
            "model_id": args.model,
            "revision": args.model_revision,
            "dtype": args.dtype,
            "device": str(device),
            "filter_special_tokens": not args.allow_special_tokens,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
        },
        "execution": {
            "physical_gpu_id": os.environ.get("EXPERIMENT_PHYSICAL_GPU_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    atomic_write_json(metadata_path, metadata)

    try:
        adapter = DreamAdapter.from_pretrained(
            ModelLoadOptions(
                model_id=args.model,
                revision=args.model_revision,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
                dtype=dtype,
                device=device,
            ),
            filter_special_tokens=not args.allow_special_tokens,
        )
        hardware = hardware_description(torch, device)
        metadata["hardware"] = hardware
        atomic_write_json(metadata_path, metadata)

        batches = prepare_batches(
            adapter,
            torch,
            records,
            batch_size=args.batch_size,
            continuation_length=args.continuation_length,
        )

        determinism: dict[str, bool] = {}
        for k in args.k_values:
            print(f"[k={k}] warm-up", flush=True)
            warm_up(
                adapter,
                torch,
                device,
                batches,
                k=k,
                warmup_batches=args.warmup_batches,
            )
            canonical_ids = None
            for repeat in range(args.timing_repeats):
                timing, generated = time_schedule(
                    adapter,
                    torch,
                    device,
                    batches,
                    k=k,
                    timing_repeat=repeat,
                    retain_outputs=repeat == 0,
                )
                timing.update(
                    {
                        "batch_size": args.batch_size,
                        "hardware": hardware,
                        "precision": args.dtype,
                    }
                )
                append_jsonl(timing_path, timing)
                print(
                    f"[k={k} repeat={repeat}] "
                    f"{timing['elapsed_seconds']:.3f}s total; "
                    f"{timing['amortized_seconds_per_sample']:.6f}s/sample",
                    flush=True,
                )
                if generated is not None:
                    canonical_ids = generated

            if canonical_ids is None:
                raise RuntimeError("Canonical generation corpus was not retained.")
            generation_path = output_dir / "generations" / f"k_{k}.jsonl"
            save_generations(
                generation_path,
                adapter.tokenizer,
                records,
                canonical_ids,
                k=k,
                continuation_length=args.continuation_length,
                mask_token_id=adapter.mask_token_id,
            )
            determinism[str(k)] = verify_determinism(
                adapter,
                torch,
                records,
                canonical_ids,
                k=k,
                example_count=args.determinism_check_examples,
                continuation_length=args.continuation_length,
                batch_size=args.batch_size,
            )
            if not determinism[str(k)]:
                raise RuntimeError(f"Determinism check failed for k={k}.")

        print("[trajectories] collecting untimed traces", flush=True)
        trajectory_rows = collect_trajectories(
            adapter,
            torch,
            records,
            args.trajectory_example_ids,
            k_values=args.k_values,
            continuation_length=args.continuation_length,
        )
        atomic_write_jsonl(output_dir / "trajectories.jsonl", trajectory_rows)

        metadata["status"] = "complete"
        metadata["completed_at_utc"] = utc_now()
        metadata["determinism_checks"] = determinism
        metadata["artifacts"] = {
            "timings": "timings.jsonl",
            "trajectories": "trajectories.jsonl",
            "generations": {
                str(k): f"generations/k_{k}.jsonl" for k in args.k_values
            },
        }
        atomic_write_json(metadata_path, metadata)
        print(f"[done] run directory: {output_dir}", flush=True)
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["failed_at_utc"] = utc_now()
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(metadata_path, metadata)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
