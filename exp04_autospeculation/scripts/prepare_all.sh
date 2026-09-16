#!/usr/bin/env bash
# Freeze the four benchmarks.
#
# gsm8k and mbpp_plus are NOT regenerated when they carry a PORTED_FROM.txt:
# those were copied verbatim from the Dream run and their sha256 verified
# against that run's recorded dataset_sha256, so that --sample-size 100 --seed
# 271828 selects the same 100 tasks the reference frontier was measured on.
# Regenerating them would risk a different sample and silently unpair every
# comparison against that frontier. Set FORCE_REGENERATE=1 to override, which
# you should only do if you intend to discard the ported frontier.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
SEED="${SEED:-12345}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

prepare() {
    local name="$1"; shift
    if [ -f "$DATA_ROOT/$name/PORTED_FROM.txt" ] && [ "${FORCE_REGENERATE:-0}" != "1" ]; then
        echo "SKIP $name: ported from the Dream run, task pairing is locked."
        echo "     $(sed -n 2p "$DATA_ROOT/$name/PORTED_FROM.txt")"
        return
    fi
    echo "PREPARE $name"
    "$@"
}

prepare mbpp_plus      "$PYTHON_BIN" scripts/prepare_mbpp_plus.py      --output-dir "$DATA_ROOT/mbpp_plus" --seed "$SEED"
prepare gsm8k          "$PYTHON_BIN" scripts/prepare_gsm8k.py          --output-dir "$DATA_ROOT/gsm8k" --seed "$SEED"
prepare humaneval_plus "$PYTHON_BIN" scripts/prepare_humaneval_plus.py --output-dir "$DATA_ROOT/humaneval_plus" --seed "$SEED"
prepare math_500       "$PYTHON_BIN" scripts/prepare_math500.py        --output-dir "$DATA_ROOT/math_500" --seed "$SEED"
echo "Frozen benchmark artifacts: $DATA_ROOT"
