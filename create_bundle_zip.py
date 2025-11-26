"""
Utility to bundle the XR/IR/UV tools into a single zip archive.

Default contents:
- README.md
- src/ (all source files)
- xr_cli.spec

Usage:
    python create_bundle_zip.py --output xr_tools_bundle.zip
"""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INCLUDE = [
    "README.md",
    "src",
    "xr_cli.spec",
]
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "dist",
    "build",
    ".venv",
    "runs",
    ".mypy_cache",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}


def should_skip(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def add_path(zip_file: zipfile.ZipFile, path: Path, base: Path) -> None:
    if path.is_file():
        arcname = path.relative_to(base)
        zip_file.write(path, arcname)
        return

    for file_path in path.rglob("*"):
        if file_path.is_file() and not should_skip(file_path):
            arcname = file_path.relative_to(base)
            zip_file.write(file_path, arcname)


def create_bundle(output: Path, include: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for rel_path in include:
            path = BASE_DIR / rel_path
            if not path.exists():
                raise FileNotFoundError(f"Pfad nicht gefunden: {path}")
            add_path(zip_file, path, BASE_DIR)
    print(f"Zip-Archiv erstellt: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Erzeuge ein Zip-Archiv der XR-Tools.")
    parser.add_argument(
        "--output",
        default="xr_tools_bundle.zip",
        help="Zielpfad des Zip-Archivs (Standard: xr_tools_bundle.zip)",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=DEFAULT_INCLUDE,
        help="Relative Pfade/Dateien, die in das Archiv aufgenommen werden sollen.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    create_bundle(output_path, args.include)


if __name__ == "__main__":
    main()
