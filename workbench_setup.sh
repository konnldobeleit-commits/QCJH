#!/usr/bin/env bash
set -euo pipefail

# Simple setup script for NVIDIA AI Workbench sessions.
# - Installs Python deps that are not already bundled with the NVIDIA PyTorch base image
# - Verifies GPU visibility and prints a quick smoke-test command hint

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${THIS_DIR}/requirements_workbench.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "requirements_workbench.txt not found next to workbench_setup.sh" >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r "${REQ_FILE}"

# Optional: ensure CUDA is visible to PyTorch
python - <<'PY'
import torch
print("CUDA verfügbar:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU-Name:", torch.cuda.get_device_name(0))
PY

echo "\nSetup abgeschlossen. Beispiel-Kommandos:" 
echo "  python src/xr_cli.py tile --xr_path data/raw/P0279RoentgenGesamt.bmp --ir_path data/raw/P0279_Infrarot_gesamt.tif --uv_path data/raw/P0279_UV_gesamt.tif --out_dir data/patches/train/normal"
echo "  python src/xr_cli.py train --data_dir data/patches/train/normal --num_epochs 2 --batch_size 4"
