import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_xriu import XRIUVDataset
from models_draem_xr import DRAEM_XR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def anomaly_generator(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Erzeuge einfache synthetische Anomalien (Noise + rechteckige Maske)."""
    batch, channels, height, width = x.shape
    texture = torch.randn_like(x) * 0.3 + 0.5
    mask = torch.zeros((batch, 1, height, width), device=x.device)

    for i in range(batch):
        y0 = random.randint(0, height // 2)
        x0 = random.randint(0, width // 2)
        h = random.randint(height // 8, height // 3)
        w = random.randint(width // 8, width // 3)
        mask[i, 0, y0 : y0 + h, x0 : x0 + w] = 1.0

    beta = torch.rand(batch, 1, 1, 1, device=x.device) * 0.8 + 0.2
    anomalous = x * (1 - mask) + beta * texture * mask + (1 - beta) * x * mask
    return anomalous, mask


def train(
    data_dir: str = "data/patches/train/normal",
    num_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> None:
    dataset = XRIUVDataset(image_dir=data_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    model = DRAEM_XR(in_ch=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    reconstruction_loss = nn.L1Loss()
    segmentation_loss = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            images = batch["image"].to(device)
            noisy_images, masks = anomaly_generator(images)

            optimizer.zero_grad()
            reconstruction, segmentation = model(noisy_images, images)

            loss_recon = reconstruction_loss(reconstruction, images)
            loss_segm = segmentation_loss(segmentation, masks)
            loss = loss_recon + loss_segm

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
            f"(recon={loss_recon.item():.4f}, segm={loss_segm.item():.4f})"
        )

        checkpoint_dir = Path("runs/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict()},
            checkpoint_dir / f"xr_draem_epoch{epoch:03d}.pt",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trainiere das DRAEM-XRIUV-Modell")
    parser.add_argument("--data_dir", default="data/patches/train/normal", help="Pfad zu den Trainingspatches")
    parser.add_argument("--num_epochs", type=int, default=50, help="Anzahl der Trainingsepochen")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch-Größe")
    parser.add_argument("--lr", type=float, default=1e-4, help="Lernrate")

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
