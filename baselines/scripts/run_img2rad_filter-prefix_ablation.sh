#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="configs/img2rad.yaml"
TMP_CONFIG_DIR="configs/generated/filterablations"
mkdir -p "$TMP_CONFIG_DIR"

PYTHON_BIN="python"

COMBINATIONS=(
  "original_"
  "wavelet-"
  "original_|wavelet-"
  "original_|logarithm_"
  "original_|squareroot_"
  "original_|square_"
  "original_|exponential_"
  "original_|log-sigma-"
  "original_|wavelet-|log-sigma-"
  "original_|wavelet-|square_|squareroot_|logarithm_|exponential_|log-sigma-"
)

for combo in "${COMBINATIONS[@]}"; do
  tag="${combo//|/_}"
  tag="${tag//-/_}"
  tag="${tag//./_}"

  new_config="${TMP_CONFIG_DIR}/img2rad_${tag}_fold0.yaml"

  echo "=================================================="
  echo "Generating config: $new_config"
  echo "Prefixes: $combo"
  echo "Fold: 0"
  echo "=================================================="

  $PYTHON_BIN - <<PY
import yaml
from pathlib import Path

base_config_path = Path("${BASE_CONFIG}")
new_config_path = Path("${new_config}")
combo = "${combo}".split("|")
tag = "${tag}"

with open(base_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# prefix 조합 변경
cfg["data"]["radiomics_valid_prefixes"] = combo

# fold 0만 수행
cfg["runtime"]["folds"] = [0]

# 출력 경로 분리
cfg["paths"]["checkpoint_dir"] = f"/root/workspace/RaPaCL/outputs/img2rad/filterablations/checkpoints-{tag}-fold0"
cfg["paths"]["log_dir"] = f"/root/workspace/RaPaCL/outputs/img2rad/filterablations/logs-{tag}-fold0"

with open(new_config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print(f"[OK] wrote: {new_config_path}")
PY

  echo "Running training with $new_config"
  $PYTHON_BIN -m src.baselines.img2rad.main \
    --config "$new_config" \
    --mode train \
    --batch_size 32 
done


# run: 
# chmod +x scripts/run_img2rad_filter-prefix_ablation.sh
# ./scripts/run_img2rad_prefix_filter-ablation.sh