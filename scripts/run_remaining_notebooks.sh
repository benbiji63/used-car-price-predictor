#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for nb in 03_xgboost 04_lightgbm 05_catboost 06_mlp 07_model_comparison; do
  echo "=== Executing $nb $(date -Iseconds) ==="
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "notebooks/${nb}.ipynb"
  echo "=== Finished $nb $(date -Iseconds) ==="
done
