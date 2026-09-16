#!/usr/bin/env bash
# Build the fixed OpenWebText evaluation dataset once, in the foreground.
#
# Example override:
#   PREFIX_LENGTH=96 CONTINUATION_LENGTH=192 bash build_owt_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# This cluster may inherit a virtual-memory cap that makes pandas/pyarrow/scipy
# abort while creating native worker threads, even when physical RAM is free.
ulimit -v unlimited

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export ARROW_NUM_THREADS="${ARROW_NUM_THREADS:-4}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BUILDER="${BUILDER:-$SCRIPT_DIR/build_owt_dataset.py}"

NUM_EXAMPLES="${NUM_EXAMPLES:-5000}"
PREFIX_LENGTH="${PREFIX_LENGTH:-64}"
CONTINUATION_LENGTH="${CONTINUATION_LENGTH:-128}"
HELDOUT_SEED="${HELDOUT_SEED:-12345}"
HELDOUT_FRACTION="${HELDOUT_FRACTION:-0.01}"

DATASET_NAME="${DATASET_NAME:-Skylion007/openwebtext}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_REVISION="${DATASET_REVISION:-main}"
MODEL_ID="${MODEL_ID:-Dream-org/Dream-v0-Base-7B}"
MODEL_REVISION="${MODEL_REVISION:-}"
CACHE_DIR="${CACHE_DIR:-}"

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
DATASET_STEM="owt_n${NUM_EXAMPLES}_p${PREFIX_LENGTH}_l${CONTINUATION_LENGTH}_seed${HELDOUT_SEED}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/${DATASET_STEM}.jsonl}"
METADATA_FILE="${METADATA_FILE:-$DATA_DIR/${DATASET_STEM}.metadata.json}"

for INTEGER_NAME in NUM_EXAMPLES PREFIX_LENGTH CONTINUATION_LENGTH HELDOUT_SEED; do
    INTEGER_VALUE="${!INTEGER_NAME}"
    if ! [[ "$INTEGER_VALUE" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $INTEGER_NAME must be a nonnegative integer; received '$INTEGER_VALUE'." >&2
        exit 1
    fi
done

if [ ! -f "$BUILDER" ]; then
    echo "ERROR: dataset builder not found: $BUILDER" >&2
    exit 1
fi

if [ -f "$DATASET_FILE" ] && [ -f "$METADATA_FILE" ]; then
    echo "[already built] Reusing the fixed evaluation dataset:"
    echo "  dataset:  $DATASET_FILE"
    echo "  metadata: $METADATA_FILE"
    exit 0
fi

if [ -e "$DATASET_FILE" ] || [ -e "$METADATA_FILE" ]; then
    echo "ERROR: only one expected output exists; refusing to rebuild over a partial pair." >&2
    echo "  dataset:  $DATASET_FILE" >&2
    echo "  metadata: $METADATA_FILE" >&2
    echo "Inspect, rename, or remove the partial output before retrying." >&2
    exit 1
fi

mkdir -p "$DATA_DIR"

if ! "$PYTHON_BIN" -c 'import datasets, transformers' 2>/dev/null; then
    echo "ERROR: '$PYTHON_BIN' cannot import both datasets and transformers." >&2
    echo "Activate the Dream experiment environment, then rerun this launcher." >&2
    exit 1
fi

COMMAND=(
    "$PYTHON_BIN" -u "$BUILDER"
    --output "$DATASET_FILE"
    --metadata-output "$METADATA_FILE"
    --num-examples "$NUM_EXAMPLES"
    --prefix-length "$PREFIX_LENGTH"
    --continuation-length "$CONTINUATION_LENGTH"
    --heldout-seed "$HELDOUT_SEED"
    --heldout-fraction "$HELDOUT_FRACTION"
    --dataset-name "$DATASET_NAME"
    --dataset-split "$DATASET_SPLIT"
    --dataset-revision "$DATASET_REVISION"
    --model "$MODEL_ID"
)

if [ -n "$MODEL_REVISION" ]; then
    COMMAND+=(--model-revision "$MODEL_REVISION")
fi
if [ -n "$CACHE_DIR" ]; then
    COMMAND+=(--cache-dir "$CACHE_DIR")
fi

echo "Building the fixed Dream/OpenWebText evaluation dataset..."
echo "  examples             : $NUM_EXAMPLES"
echo "  prefix length (P)    : $PREFIX_LENGTH"
echo "  continuation length  : $CONTINUATION_LENGTH"
echo "  model input length   : $((1 + PREFIX_LENGTH + CONTINUATION_LENGTH))"
echo "  held-out seed        : $HELDOUT_SEED"
echo "  held-out fraction    : $HELDOUT_FRACTION"
echo "  dataset revision     : $DATASET_REVISION"
echo "  tokenizer model      : $MODEL_ID"
echo "  output               : $DATASET_FILE"
echo ""

"${COMMAND[@]}"

echo ""
echo "[done] Keep both files unchanged and reuse them for every k schedule."
