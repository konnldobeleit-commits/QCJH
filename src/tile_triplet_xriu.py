from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("F")
    arr = np.array(image).astype("float32")
    arr = (arr - arr.min()) / max(1e-6, (arr.max() - arr.min()))
    return arr


def tile_triplet(
    xr_path: str,
    ir_path: str,
    uv_path: str,
    out_dir: str,
    patch_size: int = 512,
    overlap: int = 256,
) -> None:
    xr_path = Path(xr_path)
    ir_path = Path(ir_path)
    uv_path = Path(uv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xr = load_gray(xr_path)
    ir = load_gray(ir_path)
    uv = load_gray(uv_path)

    if not (xr.shape == ir.shape == uv.shape):
        raise ValueError(
            f"Shape mismatch: XR {xr.shape}, IR {ir.shape}, UV {uv.shape}"
        )

    height, width = xr.shape
    step = patch_size - overlap
    patch_id = 0

    for y in range(0, height, step):
        for x in range(0, width, step):
            xr_patch = xr[y : y + patch_size, x : x + patch_size]
            ir_patch = ir[y : y + patch_size, x : x + patch_size]
            uv_patch = uv[y : y + patch_size, x : x + patch_size]

            patch_height, patch_width = xr_patch.shape
            if patch_height < patch_size or patch_width < patch_size:
                pad_h = patch_size - patch_height
                pad_w = patch_size - patch_width
                xr_patch = np.pad(xr_patch, ((0, pad_h), (0, pad_w)), mode="constant")
                ir_patch = np.pad(ir_patch, ((0, pad_h), (0, pad_w)), mode="constant")
                uv_patch = np.pad(uv_patch, ((0, pad_h), (0, pad_w)), mode="constant")

            stacked = np.stack([xr_patch, ir_patch, uv_patch], axis=-1)
            patch_uint8 = (stacked * 255).clip(0, 255).astype("uint8")
            patch_img = Image.fromarray(patch_uint8, mode="RGB")

            filename = f"hoppner_xriu_y{y:05d}_x{x:05d}.png"
            patch_img.save(out_dir / filename)
            patch_id += 1

    print(f"{patch_id} 3-Kanal-Patches erzeugt in {out_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Tile XR/IR/UV Vollbilder in 3-Kanal-Patches")
    parser.add_argument("--xr_path", default="data/raw/P0279RoentgenGesamt.bmp", help="Pfad zum XR-Bild")
    parser.add_argument("--ir_path", default="data/raw/P0279_Infrarot_gesamt.tif", help="Pfad zum IR-Bild")
    parser.add_argument("--uv_path", default="data/raw/P0279_UV_gesamt.tif", help="Pfad zum UV-Bild")
    parser.add_argument("--out_dir", default="data/patches/train/normal", help="Ausgabeverzeichnis für Patches")
    parser.add_argument("--patch_size", type=int, default=512, help="Patchgröße in Pixeln")
    parser.add_argument("--overlap", type=int, default=256, help="Überlappung in Pixeln")

    args = parser.parse_args()
    tile_triplet(
        xr_path=args.xr_path,
        ir_path=args.ir_path,
        uv_path=args.uv_path,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
