#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
SEED="${SEED:-12345}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/prepare_mbpp_plus.py --output-dir "$DATA_ROOT/mbpp_plus" --seed "$SEED"
"$PYTHON_BIN" scripts/prepare_gsm8k.py --output-dir "$DATA_ROOT/gsm8k" --seed "$SEED"
"$PYTHON_BIN" scripts/prepare_schema_json.py --output-dir "$DATA_ROOT/schema_json" --seed "$SEED"
echo "Frozen benchmark artifacts: $DATA_ROOT"
