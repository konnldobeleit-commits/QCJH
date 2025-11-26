from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class XRIUVDataset(Dataset):
    """
    Lädt 3-Kanal-Patches (XR, IR, UV) aus PNGs und optional
    korrespondierende Binärmasken.
    """

    def __init__(self, image_dir: str, mask_dir: Optional[str] = None) -> None:
        self.image_paths = sorted(list(Path(image_dir).glob("*.png")))
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def _load_mask(self, img_path: Path) -> Optional[torch.Tensor]:
        if self.mask_dir is None:
            return None
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            return None
        mask = Image.open(mask_path).convert("L")
        arr = (np.array(mask) > 0).astype("float32")
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        image = self._load_image(img_path)
        gt_mask = self._load_mask(img_path)
        return {
            "image": image,
            "gt_mask": gt_mask,
            "path": str(img_path),
        }
