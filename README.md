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

## Windows-Executable (xr_tools.exe)
Du kannst die wichtigsten Funktionen (Tiling/Training/Validierung) als `.exe` bündeln. Auf einem Windows-System mit installiertem Python 3.10+ gehst du wie folgt vor:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio scikit-learn matplotlib pillow numpy pyinstaller

# Im Repository-Hauptverzeichnis
pyinstaller xr_cli.spec --clean
```

Nach dem Build liegt `dist/xr_tools/xr_tools.exe`. Beispiele für die Nutzung (PowerShell oder CMD):

```bash
# Tiling
dist\xr_tools\xr_tools.exe tile ^
  --xr_path data\raw\P0279RoentgenGesamt.bmp ^
  --ir_path data\raw\P0279_Infrarot_gesamt.tif ^
  --uv_path data\raw\P0279_UV_gesamt.tif ^
  --out_dir data\patches\train\normal ^
  --patch_size 512 --overlap 256

# Training
dist\xr_tools\xr_tools.exe train ^
  --data_dir data\patches\train\normal ^
  --num_epochs 50 --batch_size 16

# Validierung
dist\xr_tools\xr_tools.exe validate ^
  --img_dir data\patches\val\images ^
  --mask_dir data\patches\val\masks ^
  --checkpoint_path runs\checkpoints\xr_draem_epoch049.pt
```

Hinweise:
- Das gebaute Binary nutzt CUDA nur, wenn auf dem Windows-System der passende NVIDIA-Treiber und eine kompatible PyTorch-Build installiert sind.
- Das Packaging muss auf Windows erfolgen (PyInstaller unterstützt kein Cross-Compile von Linux nach Windows).

## NVIDIA AI Workbench (GPU-Workflow)
So bekommst du das Toolkit in einer Workbench-Session lauffähig:

1. **Projekt öffnen/importieren**: Repository in Workbench als Projekt einbinden. Als Basis-Container eignet sich z. B. `nvcr.io/nvidia/pytorch:24.07-py3` (enthält bereits PyTorch + CUDA).
2. **Abhängigkeiten installieren**: Im Projekt-Container einmalig das Setup-Skript ausführen:
   ```bash
   ./workbench_setup.sh
   ```
   Dadurch werden die zusätzlichen Python-Pakete aus `requirements_workbench.txt` installiert und ein kurzer CUDA-Sanity-Check ausgegeben.
3. **Datenspeicher bereitstellen**: Lege deine Vollbilder in `data/raw/` ab. Stelle sicher, dass der Workbench-Container auf das Verzeichnis (lokal oder via „Datasets“) Zugriff hat.
4. **Pipeline ausführen**: Innerhalb der Workbench-Shell kannst du direkt den CLI-Einstiegspunkt nutzen, z. B.:
   ```bash
   # Tiling
   python src/xr_cli.py tile \
     --xr_path data/raw/P0279RoentgenGesamt.bmp \
     --ir_path data/raw/P0279_Infrarot_gesamt.tif \
     --uv_path data/raw/P0279_UV_gesamt.tif \
     --out_dir data/patches/train/normal

   # Kurzes Smoke-Test-Training (kleine Epochen- und Batch-Size-Werte für den ersten Lauf)
   python src/xr_cli.py train --data_dir data/patches/train/normal --num_epochs 2 --batch_size 4

   # Validierung mit manuellen Masken
   python src/xr_cli.py validate \
     --img_dir data/patches/val/images \
     --mask_dir data/patches/val/masks \
     --checkpoint_path runs/checkpoints/xr_draem_epoch049.pt
   ```

Tipps für Workbench:
- Wenn du persistenten Speicher nutzt, binde `data/` als Dataset/Lokales Verzeichnis ein, damit Checkpoints und Heatmaps erhalten bleiben.
- Falls du ein Windows-Hostsystem hast, kannst du weiterhin per PyInstaller eine `.exe` bauen, brauchst das aber für Workbench nicht – dort läuft alles im Container.

## Zip-Archiv zum Weitergeben
Wenn du das komplette Projekt (Quellcode, CLI-Spec, Dokumentation) als Zip-Archiv bereitstellen möchtest, kannst du es direkt generieren:

```bash
python create_bundle_zip.py --output xr_tools_bundle.zip
```

Standardmäßig landen `README.md`, das gesamte `src/`-Verzeichnis und die `xr_cli.spec` im Archiv. Über `--include` kannst du zusätzliche Pfade anhängen. Das erzeugte Archiv liegt anschließend im Repository-Hauptordner und lässt sich weitergeben oder auf ein Zielsystem kopieren.
