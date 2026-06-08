# ─────────────────────────────────────────────────────────────────────────────
#  utils/metrics.py
#  Pixel-level segmentation metrics for binary lane detection.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import torch
import numpy as np
from collections import defaultdict


def compute_metrics(
    pred_mask: torch.Tensor | np.ndarray,
    gt_mask:   torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    smooth:    float = 1e-6,
) -> dict:
    """
    Compute binary segmentation metrics for a single prediction-GT pair.

    Parameters
    ----------
    pred_mask : (H, W) float tensor/array — probabilities or logits
    gt_mask   : (H, W) int/float array   — binary ground truth {0, 1}
    threshold : float  — binarisation threshold for pred_mask
    smooth    : float  — small value to avoid division by zero

    Returns
    -------
    dict with keys: iou, dice, precision, recall, f1, tp, fp, fn, tn
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    pred_bin = (pred_mask >= threshold).astype(np.uint8).flatten()
    gt_bin   = gt_mask.astype(np.uint8).flatten()

    tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
    tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))

    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = (2 * precision * recall) / (precision + recall + smooth)

    return {
        "iou":       float(iou),
        "dice":      float(dice),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "tn":        tn,
    }


def compute_metrics_batch(
    pred_batch: torch.Tensor,
    gt_batch:   torch.Tensor,
    threshold:  float = 0.5,
    smooth:     float = 1e-6,
) -> dict:
    """
    Compute mean metrics over a batch.

    Parameters
    ----------
    pred_batch : (B, H, W) or (B, 1, H, W) — probabilities
    gt_batch   : (B, H, W)                  — binary ground truth
    """
    if pred_batch.dim() == 4:
        pred_batch = pred_batch.squeeze(1)

    results = defaultdict(list)
    B = pred_batch.size(0)

    for i in range(B):
        m = compute_metrics(pred_batch[i], gt_batch[i], threshold, smooth)
        for k, v in m.items():
            results[k].append(v)

    return {k: float(np.mean(v)) for k, v in results.items()}


class MetricTracker:
    """
    Running accumulator for epoch-level metrics.

    Usage
    -----
        tracker = MetricTracker()
        for batch in loader:
            metrics = compute_metrics_batch(preds, gts)
            tracker.update(metrics)
        epoch_means = tracker.compute()
        tracker.reset()
    """

    def __init__(self):
        self._sums:   dict = defaultdict(float)
        self._counts: dict = defaultdict(int)

    def update(self, metrics: dict, n: int = 1) -> None:
        for k, v in metrics.items():
            self._sums[k]   += float(v) * n
            self._counts[k] += n

    def compute(self) -> dict:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def __repr__(self) -> str:
        m = self.compute()
        parts = [f"{k}={v:.4f}" for k, v in m.items()]
        return "MetricTracker(" + ", ".join(parts) + ")"


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Pretty-print metrics dict for logging."""
    items = []
    for k in ["iou", "dice", "precision", "recall", "f1"]:
        if k in metrics:
            items.append(f"{k}={metrics[k]:.4f}")
    return (prefix + " " if prefix else "") + " | ".join(items)
