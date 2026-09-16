#!/usr/bin/env bash
# Rebuild the report from outputs/final/. CPU-only; nothing here re-runs a
# benchmark. The trajectory re-run's artifacts are checked in under data/.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd .. && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/diffusion/bin/python}"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN=python3
TECTONIC="${TECTONIC:-$HOME/miniconda3/envs/diffusion/bin/tectonic}"
# The tokenizer (prompt lengths, token strings) needs the SDAR transformers overlay.
export PYTHONPATH="$PROJECT_DIR/vendor_env/tf452:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" make_figures.py
"$PYTHON_BIN" make_tables.py
"$PYTHON_BIN" make_examples.py
"$PYTHON_BIN" make_trajectory.py
"$TECTONIC" report.tex
echo "built: $PWD/report.pdf"
