#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python3}"
target="."
if [[ "${1:-}" == "--baselines" ]]; then
  target=".[baselines]"
elif [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/setup.sh [--baselines]" >&2
  exit 2
fi
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install -e "$target"
.venv/bin/python experiment.py doctor --offline
.venv/bin/python experiment.py smoke
echo "Ready. Activate with: source .venv/bin/activate"
