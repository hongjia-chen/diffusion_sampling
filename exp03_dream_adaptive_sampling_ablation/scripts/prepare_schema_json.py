#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from dream_ablation.io import sha256_file, write_json, write_jsonl


FIRST_NAMES = ["Amina", "Boris", "Chen", "Dalia", "Elena", "Farid", "Grace", "Hugo"]
LAST_NAMES = ["Park", "Singh", "Diaz", "Nwosu", "Klein", "Rossi", "Bauer", "Kim"]
CITIES = ["Austin", "Boston", "Chicago", "Denver", "Madison", "Phoenix", "Seattle"]
COLORS = ["amber", "blue", "green", "indigo", "red", "silver", "violet"]
TAGS = ["alpha", "beta", "delta", "rapid", "stable", "urgent"]


SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["person", "order", "active", "tags"],
    "properties": {
        "person": {
            "type": "object",
            "additionalProperties": False,
            "required": ["full_name", "city"],
            "properties": {
                "full_name": {"type": "string"},
                "city": {"type": "string"},
            },
        },
        "order": {
            "type": "object",
            "additionalProperties": False,
            "required": ["item", "quantity", "unit_price"],
            "properties": {
                "item": {"type": "string", "enum": COLORS},
                "quantity": {"type": "integer", "minimum": 1, "maximum": 20},
                "unit_price": {"type": "number", "minimum": 1, "multipleOf": 0.25},
            },
        },
        "active": {"type": "boolean"},
        "tags": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "uniqueItems": True,
            "items": {"type": "string", "enum": TAGS},
        },
    },
}


def make_record(rng: random.Random, split: str, index: int) -> dict[str, Any]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city = rng.choice(CITIES)
    item = rng.choice(COLORS)
    quantity = rng.randint(1, 20)
    unit_price = rng.randint(4, 80) / 4
    active = bool(rng.getrandbits(1))
    tags = rng.sample(TAGS, 2)
    target = {
        "person": {"full_name": f"{first} {last}", "city": city},
        "order": {"item": item, "quantity": quantity, "unit_price": unit_price},
        "active": active,
        "tags": tags,
    }
    facts = (
        f"First name: {first}. Last name: {last}. City: {city}. "
        f"Order item: {item}. Quantity: {quantity}. Unit price: {unit_price:.2f}. "
        f"Active: {str(active).lower()}. Tags in order: {tags[0]}, {tags[1]}."
    )
    prompt = (
        "Convert the source facts into exactly one JSON object satisfying the schema. "
        "Return JSON only, with no Markdown or explanation.\n\n"
        f"Source facts:\n{facts}\n\n"
        f"JSON Schema:\n{json.dumps(SCHEMA, sort_keys=True)}"
    )
    return {
        "benchmark": "schema_json",
        "split": split,
        "example_id": f"schema-{split}-{index:04d}",
        "prompt": prompt,
        "target": target,
        "schema": SCHEMA,
    }


def build_split(path: Path, *, seed: int, split: str, count: int) -> None:
    rng = random.Random(seed)
    rows = [make_record(rng, split, index) for index in range(count)]
    write_jsonl(path, rows)
    write_json(
        path.with_suffix(".metadata.json"),
        {"benchmark": "schema_json", "split": split, "seed": seed, "count": count,
         "sha256": sha256_file(path), "generator_schema_version": 1},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_split(args.output_dir / "dev.jsonl", seed=args.seed, split="dev", count=args.dev_size)
    build_split(
        args.output_dir / "test.jsonl",
        seed=args.seed + 1,
        split="test",
        count=args.test_size,
    )


if __name__ == "__main__":
    main()

