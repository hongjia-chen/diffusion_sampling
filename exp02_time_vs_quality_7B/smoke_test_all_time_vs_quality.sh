#!/usr/bin/env bash
# End-to-end smoke test for fixed + adaptive generation and downstream outputs.
#
# This runs the real Dream model on a tiny truncated dataset. It uses clearly
# marked synthetic MAUVE values only to test summary and plotting plumbing.
#
# Required:
#   DATASET=/absolute/path/to/full.jsonl \
#   DATASET_METADATA=/absolute/path/to/full.metadata.json \
#   bash smoke_test_all_time_vs_quality.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ulimit -v unlimited
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
SOURCE_DATASET="${SOURCE_DATASET:-${DATASET:-$SCRIPT_DIR/data/owt_p64_l128.jsonl}}"
SOURCE_METADATA="${SOURCE_METADATA:-${DATASET_METADATA:-$SCRIPT_DIR/data/owt_p64_l128.metadata.json}}"
SMOKE_NUM_EXAMPLES="${SMOKE_NUM_EXAMPLES:-2}"
SMOKE_PREFIX_LENGTH="${SMOKE_PREFIX_LENGTH:-8}"
SMOKE_CONTINUATION_LENGTH="${SMOKE_CONTINUATION_LENGTH:-8}"
SMOKE_ROOT="${SMOKE_ROOT:-$SCRIPT_DIR/outputs/time_vs_quality_smoke}"
SMOKE_DATASET="$SMOKE_ROOT/data/smoke.jsonl"
SMOKE_METADATA="$SMOKE_ROOT/data/smoke.metadata.json"
FIXED_RUN="$SMOKE_ROOT/fixed"
ADAPTIVE_RUN="$SMOKE_ROOT/adaptive"
COMPARISON_DIR="$SMOKE_ROOT/comparison"
MODEL_ID="${MODEL_ID:-Dream-org/Dream-v0-Base-7B}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

for path in "$SOURCE_DATASET" "$SOURCE_METADATA"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: source dataset artifact is missing: $path" >&2
        exit 1
    fi
done

REQUIRED_FILES=(
    dream_adapter.py
    adaptive_scheduler.py
    run_dream_topk.py
    audit_dream_topk.py
    summarize_dream_topk.py
    run_dream_adaptive.py
    audit_dream_adaptive.py
    summarize_dream_adaptive.py
    extract_dream_trajectories.py
    plot_time_vs_quality.py
    test_adaptive_scheduler.py
)
for path in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: required file is missing: $path" >&2
        exit 1
    fi
done

"$PYTHON_BIN" -m py_compile "${REQUIRED_FILES[@]}"
"$PYTHON_BIN" test_adaptive_scheduler.py
for script in \
    run_dream_topk.py audit_dream_topk.py summarize_dream_topk.py \
    run_dream_adaptive.py audit_dream_adaptive.py \
    summarize_dream_adaptive.py extract_dream_trajectories.py \
    plot_time_vs_quality.py
do
    "$PYTHON_BIN" "$script" --help >/dev/null
done
echo "[OK] unit tests, syntax, and command-line interfaces"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    sys.exit("ERROR: selected GPU is not visible as one logical CUDA device.")
p = torch.cuda.get_device_properties(0)
print(f"[OK] CUDA logical device 0: {p.name} ({p.total_memory / 1024**3:.1f} GiB)")
PY

mkdir -p "$SMOKE_ROOT/data" "$COMPARISON_DIR"
"$PYTHON_BIN" - "$SOURCE_DATASET" "$SOURCE_METADATA" \
    "$SMOKE_DATASET" "$SMOKE_METADATA" \
    "$SMOKE_NUM_EXAMPLES" "$SMOKE_PREFIX_LENGTH" \
    "$SMOKE_CONTINUATION_LENGTH" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source_dataset, source_metadata, target_dataset, target_metadata = map(
    Path, sys.argv[1:5]
)
n, prefix_length, continuation_length = map(int, sys.argv[5:8])
rows = []
with source_dataset.open(encoding="utf-8") as handle:
    for line in handle:
        if len(rows) == n:
            break
        row = json.loads(line)
        row["example_id"] = len(rows)
        row["prefix_token_ids"] = row["prefix_token_ids"][:prefix_length]
        row["human_continuation_token_ids"] = (
            row["human_continuation_token_ids"][:continuation_length]
        )
        rows.append(row)
if len(rows) != n:
    raise SystemExit("Source dataset has too few records.")
with target_dataset.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
metadata["num_examples"] = n
metadata["prefix_length"] = prefix_length
metadata["continuation_length"] = continuation_length
metadata["dataset_sha256"] = hashlib.sha256(
    target_dataset.read_bytes()
).hexdigest()
target_metadata.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"[OK] smoke dataset: N={n}, P={prefix_length}, L={continuation_length}")
PY

TRAJECTORY_IDS=()
for ((example_id = 0; example_id < SMOKE_NUM_EXAMPLES; example_id++)); do
    TRAJECTORY_IDS+=("$example_id")
done

model_args=(--model "$MODEL_ID" --device cuda:0 --dtype bfloat16)
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    model_args+=(--local-files-only)
fi

echo "[smoke] real fixed-k generation"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u run_dream_topk.py \
    --dataset "$SMOKE_DATASET" \
    --dataset-metadata "$SMOKE_METADATA" \
    --output-dir "$FIXED_RUN" \
    --prefix-length "$SMOKE_PREFIX_LENGTH" \
    --continuation-length "$SMOKE_CONTINUATION_LENGTH" \
    --k-values 1 2 3 4 5 6 7 8 \
    --batch-size 1 \
    --timing-repeats 1 \
    --warmup-batches 1 \
    --trajectory-example-ids "${TRAJECTORY_IDS[@]}" \
    --determinism-check-examples "$SMOKE_NUM_EXAMPLES" \
    "${model_args[@]}" \
    --overwrite

"$PYTHON_BIN" -u audit_dream_topk.py "$FIXED_RUN" \
    --dataset "$SMOKE_DATASET" \
    --dataset-metadata "$SMOKE_METADATA" \
    --prefix-length "$SMOKE_PREFIX_LENGTH" \
    --continuation-length "$SMOKE_CONTINUATION_LENGTH" \
    --k-values 1 2 3 4 5 6 7 8

echo "[smoke] real adaptive generation"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u run_dream_adaptive.py \
    --dataset "$SMOKE_DATASET" \
    --dataset-metadata "$SMOKE_METADATA" \
    --output-dir "$ADAPTIVE_RUN" \
    --prefix-length "$SMOKE_PREFIX_LENGTH" \
    --continuation-length "$SMOKE_CONTINUATION_LENGTH" \
    --tau-multipliers 1.01 1.10 1.25 \
    --max-candidates 8 \
    --batch-size 1 \
    --conditional-microbatch-size 1 \
    --timing-repeats 1 \
    --warmup-examples 1 \
    --trajectory-example-ids "${TRAJECTORY_IDS[@]}" \
    --determinism-check-examples "$SMOKE_NUM_EXAMPLES" \
    "${model_args[@]}" \
    --overwrite

"$PYTHON_BIN" -u audit_dream_adaptive.py "$ADAPTIVE_RUN" \
    --dataset "$SMOKE_DATASET" \
    --dataset-metadata "$SMOKE_METADATA" \
    --prefix-length "$SMOKE_PREFIX_LENGTH" \
    --continuation-length "$SMOKE_CONTINUATION_LENGTH"

echo "[smoke] writing synthetic metric fixtures for summary/plot wiring"
"$PYTHON_BIN" - "$FIXED_RUN" "$ADAPTIVE_RUN" <<'PY'
import json
import sys
from pathlib import Path

fixed_run, adaptive_run = map(Path, sys.argv[1:3])
fixed = json.loads((fixed_run / "metadata.json").read_text())
adaptive = json.loads((adaptive_run / "metadata.json").read_text())
settings = {"smoke_fixture": True, "not_a_scientific_mauve_result": True}

with (fixed_run / "mauve.jsonl").open("w", encoding="utf-8") as handle:
    for k in fixed["experiment"]["k_values"]:
        row = {
            "k": k,
            "num_examples": fixed["dataset"]["num_examples"],
            "prefix_length": fixed["experiment"]["prefix_length"],
            "continuation_length": fixed["experiment"]["continuation_length"],
            "mauve": 0.50 + 0.01 * int(k),
            "frontier_integral": 0.25,
            "settings": settings,
            "dataset_sha256": fixed["dataset"]["sha256"],
        }
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")

with (adaptive_run / "adaptive_mauve.jsonl").open(
    "w", encoding="utf-8"
) as handle:
    for index, threshold in enumerate(
        adaptive["experiment"]["thresholds"]
    ):
        row = {
            "schedule_type": "adaptive",
            "threshold_label": threshold["label"],
            "tau_multiplier": threshold["multiplier"],
            "tau": threshold["tau"],
            "num_examples": adaptive["dataset"]["num_examples"],
            "prefix_length": adaptive["experiment"]["prefix_length"],
            "continuation_length": adaptive["experiment"][
                "continuation_length"
            ],
            "mauve": 0.56 + 0.01 * index,
            "frontier_integral": 0.25,
            "settings": settings,
            "dataset_sha256": adaptive["dataset"]["sha256"],
        }
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")

(adaptive_run.parent / "SMOKE_METRICS_ARE_SYNTHETIC.txt").write_text(
    "The MAUVE values under this smoke directory are synthetic fixtures.\\n",
    encoding="utf-8",
)
PY

"$PYTHON_BIN" -u summarize_dream_topk.py "$FIXED_RUN"
"$PYTHON_BIN" -u summarize_dream_adaptive.py "$ADAPTIVE_RUN"

"$PYTHON_BIN" -u extract_dream_trajectories.py \
    --fixed-run "$FIXED_RUN" \
    --adaptive-run "$ADAPTIVE_RUN" \
    --dataset "$SMOKE_DATASET" \
    --example-ids "${TRAJECTORY_IDS[@]}" \
    --output-dir "$COMPARISON_DIR/trajectories"

"$PYTHON_BIN" -u plot_time_vs_quality.py \
    --fixed-summary "$FIXED_RUN/summary.csv" \
    --adaptive-summary "$ADAPTIVE_RUN/adaptive_summary.csv" \
    --output-dir "$COMPARISON_DIR"

for expected in \
    "$FIXED_RUN/audit.json" \
    "$ADAPTIVE_RUN/audit.json" \
    "$COMPARISON_DIR/time_vs_quality.png" \
    "$COMPARISON_DIR/time_vs_quality.pdf" \
    "$COMPARISON_DIR/trajectories/trajectory_all.txt" \
    "$COMPARISON_DIR/trajectories/trajectory_steps.jsonl"
do
    if [ ! -s "$expected" ]; then
        echo "ERROR: smoke output is missing or empty: $expected" >&2
        exit 1
    fi
done

echo "[PASS] complete fixed + adaptive smoke test"
echo "  root:        $SMOKE_ROOT"
echo "  plot:        $COMPARISON_DIR/time_vs_quality.png"
echo "  trajectories: $COMPARISON_DIR/trajectories/trajectory_all.txt"
echo "  NOTE: smoke MAUVE values are synthetic fixtures, not experiment results."
