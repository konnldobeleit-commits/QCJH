# QCJH

Dieses Repository enthält Skripte für ein dreikanaliges XR/IR/UV-Anomalie-Setup nach DRAEM-Vorbild. Die folgenden Schritte zeigen, wie du das Projekt lokal ausführst.

## Voraussetzungen
- Python 3.10+
- PyTorch mit CUDA-Unterstützung (falls du eine NVIDIA-GPU nutzt)
- Optional: `scikit-learn` und `matplotlib` für die Validierungsauswertung

```bash
pip install torch torchvision torchaudio  # wähle die zu deiner CUDA-Version passende PyTorch-Variante
pip install scikit-learn matplotlib pillow numpy
```

## Daten vorbereiten (Tiling)
1. Lege deine registrierten Vollbilder in `data/raw/` ab:
   - `P0279RoentgenGesamt.bmp` (XR)
   - `P0279_Infrarot_gesamt.tif` (IR)
   - `P0279_UV_gesamt.tif` (UV)
2. Erzeuge überlappende 512×512-Patches mit synchronen Kanälen:

```bash
python src/tile_triplet_xriu.py \
  --xr_path data/raw/P0279RoentgenGesamt.bmp \
  --ir_path data/raw/P0279_Infrarot_gesamt.tif \
  --uv_path data/raw/P0279_UV_gesamt.tif \
  --out_dir data/patches/train/normal \
  --patch_size 512 --overlap 256
```

Die Patches liegen anschließend als 3-Kanal-PNGs in `data/patches/train/normal/`.

## Training (DRAEM-Variante)
1. Stelle sicher, dass `train_draem_xr.py` dein Dreikanal-Setup nutzt (Dataset `XRIUVDataset`, `DRAEM_XR(in_ch=3)`).
2. Starte das Training, z. B. mit einer RTX-GPU:

```bash
python src/train_draem_xr.py --data_dir data/patches/train/normal --num_epochs 50 --batch_size 16
```

Checkpoints landen in `runs/checkpoints/`.

## Validierung (ROC/IoU)
1. Lege für ausgewählte Patches GT-Masken in `data/patches/val/masks/` an (gleicher Dateiname wie im `images/`-Ordner).
2. Führe die Validierung aus und erhalte ROC-AUC sowie IoU-Werte:

```bash
python src/validate_draem_xriu.py \
  --img_dir data/patches/val/images \
  --mask_dir data/patches/val/masks \
  --checkpoint_path runs/checkpoints/xr_draem_epoch049.pt
```

Eine ROC-Kurve wird optional als PNG in `runs/roc_curve.png` gespeichert, wenn `matplotlib` installiert ist.

## Inferenz-Idee (Heatmaps)
Aktuell liegt ein Validierungs-Skript vor. Für ganze Heatmaps kannst du das Diskriminator-Ausgabeprinzip aus `validate_draem_xriu.py` auf überlappende Kacheln übertragen (analog zum Tiling-Schritt), um pixelweise Anomaliewahrscheinlichkeiten über das gesamte Bild zu erhalten.
