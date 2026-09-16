#!/usr/bin/env python3
"""Create a checksum-pinned first-N subset of the verified OWT corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from run_dream_topk import atomic_write_json, atomic_write_jsonl, load_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.num_examples < 1:
        parser.error("--num-examples must be positive.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    source_metadata_path = args.source_metadata.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata_output = args.metadata_output.expanduser().resolve()
    for path in (source, source_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.overwrite and (output.exists() or metadata_output.exists()):
        if output.is_file() and metadata_output.is_file():
            existing = load_json(metadata_output)
            if (
                int(existing.get("num_examples", -1)) == args.num_examples
                and existing.get("dataset_sha256") == sha256_file(output)
                and existing.get("subset", {}).get("source_dataset_sha256")
                == sha256_file(source)
            ):
                print(f"[already built] subset dataset: {output}")
                return 0
        raise FileExistsError(
            "Subset output already exists but is not the requested verified pair."
        )

    source_metadata = load_json(source_metadata_path)
    source_digest = sha256_file(source)
    if source_digest != source_metadata.get("dataset_sha256"):
        raise ValueError("Source dataset checksum does not match its metadata.")
    source_count = int(source_metadata["num_examples"])
    if args.num_examples > source_count:
        raise ValueError(
            f"Requested {args.num_examples} examples from a {source_count}-row corpus."
        )

    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if len(rows) == args.num_examples:
                break
            row = json.loads(line)
            if int(row["example_id"]) != len(rows):
                raise ValueError("Source example IDs are not contiguous from zero.")
            rows.append(row)
    if len(rows) != args.num_examples:
        raise ValueError("Source dataset ended before the requested subset size.")

    atomic_write_jsonl(output, rows)
    subset_digest = sha256_file(output)
    metadata = dict(source_metadata)
    metadata["num_examples"] = args.num_examples
    metadata["dataset_sha256"] = subset_digest
    metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["subset"] = {
        "method": "first_n_in_frozen_source_order",
        "num_examples": args.num_examples,
        "source_dataset_path": str(source),
        "source_metadata_path": str(source_metadata_path),
        "source_dataset_sha256": source_digest,
        "source_num_examples": source_count,
    }
    atomic_write_json(metadata_output, metadata)
    print(f"[done] subset dataset:  {output}")
    print(f"[done] subset metadata: {metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
