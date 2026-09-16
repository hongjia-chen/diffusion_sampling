#!/usr/bin/env python3
"""Build the fixed tokenized OpenWebText dataset for the Dream top-k experiment.

OpenWebText exposes only a training split. This builder derives a deterministic
held-out partition by hashing each unique document with ``heldout_seed`` and
retaining the configured fraction. It then saves the first N usable documents
from that partition. Reusing the resulting JSONL file guarantees that every
Dream schedule sees exactly the same tokenized prefixes and human targets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DATASET = "Skylion007/openwebtext"
DEFAULT_MODEL = "Dream-org/Dream-v0-Base-7B"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--num-examples", type=int, default=5_000)
    parser.add_argument("--prefix-length", type=int, default=64)
    parser.add_argument("--continuation-length", type=int, default=128)
    parser.add_argument("--heldout-seed", type=int, default=12_345)
    parser.add_argument(
        "--heldout-fraction",
        type=float,
        default=0.01,
        help="Fraction of unique source documents assigned to the held-out partition.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--dataset-revision",
        default="main",
        help="Hugging Face dataset revision. Use an immutable commit for production.",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--max-scanned-documents",
        type=int,
        default=0,
        help="Safety limit; zero scans until enough usable examples are found.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.num_examples < 1:
        parser.error("--num-examples must be at least 1.")
    if args.prefix_length < 1:
        parser.error("--prefix-length must be at least 1.")
    if args.continuation_length < 1:
        parser.error("--continuation-length must be at least 1.")
    if not 0.0 < args.heldout_fraction < 1.0:
        parser.error("--heldout-fraction must be strictly between 0 and 1.")
    if args.max_scanned_documents < 0:
        parser.error("--max-scanned-documents cannot be negative.")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1.")
    return args


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def document_id(text: str) -> str:
    """Return a stable content-based identifier used for deduplication."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def heldout_score(doc_id: str, seed: int) -> float:
    """Map one document deterministically into the interval [0, 1)."""

    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("ascii")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(1 << 64)


def metadata_path_for(dataset_path: Path) -> Path:
    if dataset_path.suffix:
        return dataset_path.with_suffix(".metadata.json")
    return dataset_path.with_name(dataset_path.name + ".metadata.json")


def load_tokenizer(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Install the Dream-tested environment first."
        ) from exc

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    if args.cache_dir:
        kwargs["cache_dir"] = str(args.cache_dir)
    return AutoTokenizer.from_pretrained(args.model, **kwargs)


def load_source_dataset(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required. Install it in the experiment environment."
        ) from exc

    kwargs: dict[str, Any] = {
        "split": args.dataset_split,
        "streaming": True,
        "revision": args.dataset_revision,
    }
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    if args.cache_dir:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)

    return load_dataset(args.dataset_name, **kwargs)


def encode_required_tokens(
    tokenizer: Any,
    text: str,
    required_length: int,
) -> list[int]:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=required_length,
    )
    return [int(token_id) for token_id in token_ids]


def decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def write_json_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_dataset(args: argparse.Namespace) -> tuple[Path, Path]:
    output_path = args.output.expanduser().resolve()
    metadata_path = (
        args.metadata_output.expanduser().resolve()
        if args.metadata_output
        else metadata_path_for(output_path)
    )
    if output_path == metadata_path:
        raise ValueError("Dataset and metadata paths must be different.")
    existing = [path for path in (output_path, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output: {paths}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args)
    source = load_source_dataset(args)
    source_fingerprint = getattr(source, "_fingerprint", None)

    required_length = args.prefix_length + args.continuation_length
    seen_document_ids: set[str] = set()
    scanned_documents = 0
    duplicate_documents = 0
    heldout_candidates = 0
    too_short_documents = 0
    completed_examples = 0
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

            for source_index, row in enumerate(source):
                if (
                    args.max_scanned_documents
                    and scanned_documents >= args.max_scanned_documents
                ):
                    break
                scanned_documents += 1

                text = row.get(args.text_field)
                if not isinstance(text, str) or not text.strip():
                    continue

                doc_id = document_id(text)
                if doc_id in seen_document_ids:
                    duplicate_documents += 1
                    continue
                seen_document_ids.add(doc_id)

                score = heldout_score(doc_id, args.heldout_seed)
                if score >= args.heldout_fraction:
                    continue
                heldout_candidates += 1

                token_ids = encode_required_tokens(
                    tokenizer,
                    text,
                    required_length,
                )
                if len(token_ids) < required_length:
                    too_short_documents += 1
                    continue

                prefix_ids = token_ids[: args.prefix_length]
                human_ids = token_ids[
                    args.prefix_length : args.prefix_length
                    + args.continuation_length
                ]
                record = {
                    "example_id": completed_examples,
                    "document_id": doc_id,
                    "source_index": source_index,
                    "heldout_score": score,
                    "prefix_token_ids": prefix_ids,
                    "prefix_text": decode_tokens(tokenizer, prefix_ids),
                    "human_continuation_token_ids": human_ids,
                    "human_continuation_text": decode_tokens(
                        tokenizer,
                        human_ids,
                    ),
                }
                write_json_line(handle, record)
                completed_examples += 1

                if completed_examples % args.progress_every == 0:
                    print(
                        f"[dataset] completed={completed_examples}/"
                        f"{args.num_examples} scanned={scanned_documents}",
                        flush=True,
                    )
                if completed_examples == args.num_examples:
                    break

            if completed_examples != args.num_examples:
                raise RuntimeError(
                    "Unable to build the requested dataset: "
                    f"completed={completed_examples}, requested={args.num_examples}, "
                    f"scanned={scanned_documents}, "
                    f"heldout_candidates={heldout_candidates}, "
                    f"too_short={too_short_documents}."
                )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_file": output_path.name,
        "dataset_sha256": sha256_file(output_path),
        "num_examples": args.num_examples,
        "prefix_length": args.prefix_length,
        "continuation_length": args.continuation_length,
        "model_input_length": 1 + args.prefix_length + args.continuation_length,
        "tokenizer": {
            "model_id": args.model,
            "revision": args.model_revision,
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "mask_token_id": getattr(tokenizer, "mask_token_id", None),
        },
        "source": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.dataset_split,
            "revision": args.dataset_revision,
            "fingerprint": source_fingerprint,
            "text_field": args.text_field,
        },
        "heldout_partition": {
            "method": "sha256(seed:document_sha256), first 64 bits",
            "seed": args.heldout_seed,
            "fraction": args.heldout_fraction,
            "selection": "first N usable unique documents in source order",
        },
        "build_counts": {
            "scanned_documents": scanned_documents,
            "unique_documents": len(seen_document_ids),
            "duplicate_documents": duplicate_documents,
            "heldout_candidates": heldout_candidates,
            "too_short_documents": too_short_documents,
            "completed_examples": completed_examples,
        },
        "software": {
            "python": sys.version.split()[0],
            "datasets": package_version("datasets"),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
        },
    }
    atomic_write_json(metadata_path, metadata)
    return output_path, metadata_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path, metadata_path = build_dataset(args)
    print(f"[done] dataset:  {dataset_path}")
    print(f"[done] metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
