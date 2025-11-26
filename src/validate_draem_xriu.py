from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from dataset_xriu import XRIUVDataset
from models_draem_xr import DRAEM_XR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_iou(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    pred_bin = (pred >= thr).astype("float32")
    gt_bin = (gt > 0.5).astype("float32")
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def validate(
    img_dir: str = "data/patches/val/images",
    mask_dir: str = "data/patches/val/masks",
    checkpoint_path: str = "runs/checkpoints/xr_draem_epoch049.pt",
    batch_size: int = 8,
    iou_thresholds: Iterable[float] = (0.3, 0.5, 0.7),
) -> None:
    dataset = XRIUVDataset(image_dir=img_dir, mask_dir=mask_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = DRAEM_XR(in_ch=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    iou_sums = {thr: 0.0 for thr in iou_thresholds}
    iou_counts = {thr: 0 for thr in iou_thresholds}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt = batch["gt_mask"]
            if isinstance(gt, list):
                if any(mask is None for mask in gt):
                    continue
                gt = torch.stack(gt, dim=0)
            elif gt is None:
                continue
            gt = gt.to(device)

            recon, segm = model(images, images)
            segm_np = segm.squeeze(1).cpu().numpy()
            gt_np = gt.squeeze(1).cpu().numpy()

            for batch_idx in range(segm_np.shape[0]):
                pred_b = segm_np[batch_idx]
                gt_b = gt_np[batch_idx]
                if gt_b.sum() == 0:
                    continue
                for thr in iou_thresholds:
                    iou = compute_iou(pred_b, gt_b, thr=thr)
                    iou_sums[thr] += iou
                    iou_counts[thr] += 1

            all_scores.append(segm_np.reshape(-1))
            all_labels.append(gt_np.reshape(-1))

    if not all_scores:
        print("Keine GT-Masken gefunden – ROC/IoU können nicht berechnet werden.")
        return

    y_score = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    valid = (y_true == 0) | (y_true == 1)
    y_score = y_score[valid]
    y_true = y_true[valid]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC über alle Masken-Pixel: {roc_auc:.4f}")

    for thr in iou_thresholds:
        if iou_counts[thr] > 0:
            mean_iou = iou_sums[thr] / iou_counts[thr]
            print(
                f"Durchschnittliche IoU bei Schwelle {thr:.2f}: {mean_iou:.4f} "
                f"(n={iou_counts[thr]})"
            )
        else:
            print(f"Keine Patches mit GT-Anomalie für Schwelle {thr:.2f}")

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("XR/IR/UV Anomaly Detection – ROC")
        plt.legend(loc="lower right")
        out_path = Path("runs/roc_curve.png")
        plt.savefig(out_path, dpi=200)
        print(f"ROC-Kurve gespeichert unter {out_path}")
    except ImportError:
        print("matplotlib nicht installiert – ROC-Kurve wird nur als Kennzahlen ausgegeben.")


def main() -> None:
    validate()


if __name__ == "__main__":
    main()
