#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="configs/img2rad.yaml"
GEN_CONFIG_DIR="configs/generated/archablations"
mkdir -p "$GEN_CONFIG_DIR"

PYTHON_BIN="python"

# ARCHS=("radpred" "radhidden" "rawrad")
ARCHS=("rawrad" "radhidden")

for arch in "${ARCHS[@]}"; do
  case "$arch" in
    radpred)
      FUSION_MODE="img_radpred"
      FREEZE="false"
      TAG="v0_imgRadPred"
      ;;
    radhidden)
      FUSION_MODE="img_radhidden"
      FREEZE="false"
      # FREEZE="true"
      TAG="v1_imgRadHiddenFreeze"
      ;;
    rawrad)
      FUSION_MODE="img_rawrad"
      FREEZE="false"
      TAG="v2_imgRawRad"
      ;;
    *)
      echo "Unknown arch: $arch"
      exit 1
      ;;
  esac

  NEW_CONFIG="${GEN_CONFIG_DIR}/img2rad_${TAG}_fold0.yaml"

  echo "=================================================="
  echo "Architecture: $arch"
  echo "Fusion mode : $FUSION_MODE"
  echo "Freeze      : $FREEZE"
  echo "Config      : $NEW_CONFIG"
  echo "=================================================="

  $PYTHON_BIN - <<PY
import yaml
from pathlib import Path

base_config_path = Path("${BASE_CONFIG}")
new_config_path = Path("${NEW_CONFIG}")

with open(base_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("model", {})
cfg.setdefault("runtime", {})
cfg.setdefault("paths", {})

cfg["model"]["fusion_mode"] = "${FUSION_MODE}"
cfg["model"]["freeze_img2rad"] = True if "${FREEZE}" == "true" else False
# if eval 
# cfg["model"]["radiomics_dim"] = 929

# fold 0만
cfg["runtime"]["folds"] = [0]

# 실험별 로그/체크포인트 구분
cfg["paths"]["checkpoint_dir"] = f"/root/workspace/RaPaCL/outputs/img2rad/archablations/checkpoints-${TAG}-fold0"
cfg["paths"]["log_dir"] = f"/root/workspace/RaPaCL/outputs/img2rad/archablations/logs-${TAG}-fold0"

with open(new_config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print(f"[OK] wrote {new_config_path}")
PY

  $PYTHON_BIN -m src.baselines.img2rad.main \
    --config "$NEW_CONFIG" \
    --mode all 
done


# run: 
# chmod +x scripts/run_img2rad_arch_ablation.sh
# ./scripts/run_img2rad_arch_ablation.sh