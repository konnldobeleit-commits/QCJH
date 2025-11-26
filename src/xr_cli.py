"""Kommandozeilen-Einstiegspunkt für Tiling, Training und Validierung.
Geeignet als Hauptskript für PyInstaller-Builds (xr_tools.exe).
"""
import argparse

from tile_triplet_xriu import tile_triplet
from train_draem_xr import train
from validate_draem_xriu import validate


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Wert muss > 0 sein")
    return ivalue


def cmd_tile(args: argparse.Namespace) -> None:
    tile_triplet(
        xr_path=args.xr_path,
        ir_path=args.ir_path,
        uv_path=args.uv_path,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )


def cmd_train(args: argparse.Namespace) -> None:
    train(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


def cmd_validate(args: argparse.Namespace) -> None:
    validate(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XR/IR/UV Anomaly Toolkit (Tiling, Training, Validierung)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Tile
    tile_p = subparsers.add_parser(
        "tile", help="Erzeuge 3-Kanal-Patches aus registrierten XR/IR/UV-Bildern",
    )
    tile_p.add_argument("--xr_path", required=True, help="Pfad zum XR-Vollbild")
    tile_p.add_argument("--ir_path", required=True, help="Pfad zum IR-Vollbild")
    tile_p.add_argument("--uv_path", required=True, help="Pfad zum UV-Vollbild")
    tile_p.add_argument(
        "--out_dir",
        default="data/patches/train/normal",
        help="Zielverzeichnis für Patches",
    )
    tile_p.add_argument(
        "--patch_size", type=positive_int, default=512, help="Patchgröße (px)",
    )
    tile_p.add_argument(
        "--overlap", type=int, default=256, help="Überlappung zwischen Patches",
    )
    tile_p.set_defaults(func=cmd_tile)

    # Train
    train_p = subparsers.add_parser("train", help="Trainiere das DRAEM-XRIUV-Modell")
    train_p.add_argument(
        "--data_dir",
        default="data/patches/train/normal",
        help="Pfad zu den Trainingspatches",
    )
    train_p.add_argument("--num_epochs", type=positive_int, default=50)
    train_p.add_argument("--batch_size", type=positive_int, default=16)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.set_defaults(func=cmd_train)

    # Validate
    val_p = subparsers.add_parser(
        "validate", help="ROC/IoU-Auswertung mit manuellen Masken",
    )
    val_p.add_argument(
        "--img_dir",
        default="data/patches/val/images",
        help="Verzeichnis mit Validierungs-Patches",
    )
    val_p.add_argument(
        "--mask_dir",
        default="data/patches/val/masks",
        help="Verzeichnis mit GT-Masken",
    )
    val_p.add_argument(
        "--checkpoint_path",
        default="runs/checkpoints/xr_draem_epoch049.pt",
        help="Pfad zu einem gespeicherten Checkpoint",
    )
    val_p.add_argument("--batch_size", type=positive_int, default=8)
    val_p.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
