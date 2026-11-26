#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export STARPLATE_DATA="${STARPLATE_DATA:-$PWD/data}"
export STARPLATE_HOST="${STARPLATE_HOST:-127.0.0.1}"
export PORT="${PORT:-19935}"
exec "${STARPLATE_PYTHON:-python3}" app.py
