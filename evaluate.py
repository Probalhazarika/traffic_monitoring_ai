# ─────────────────────────────────────────────────────────────────────────────
#  evaluate.py
#  Evaluation script for the aerial lane detection model.
#
#  Usage
#  ─────
#    python evaluate.py                             # uses default best model
#    python evaluate.py --checkpoint weights/best_model.pth
#    python evaluate.py --split test
#    python evaluate.py --tta                       # test-time augmentation
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.dataset     import AerialLanesDataset, verify_dataset
from utils.augmentations import get_val_transforms, get_tta_transforms
from utils.metrics     import compute_metrics, MetricTracker, format_metrics
from models            import build_model


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Aerial Lane Detection Evaluation")
    p.add_argument("--checkpoint", default="weights/best_model.pth")
    p.add_argument("--config",     default="configs/segformer_b5.yaml")
    p.add_argument("--dataset",    default="dataset")
    p.add_argument("--split",      default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--output",     default="outputs")
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--tta",        action="store_true",
                   help="Test-time augmentation (4× H/V flips)")
    p.add_argument("--vis-count",  type=int, default=20,
                   help="Number of visualization images to save")
    p.add_argument("--model",      default=None,
                   help="Override model_type: segformer or mask2former")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(model, img_tensor: torch.Tensor,
                   device: torch.device, model_type: str,
                   threshold: float) -> np.ndarray:
    """Run model on a single image tensor. Returns binary mask (H, W) uint8."""
    img = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "segformer":
            logits = model(img)                          # (1, 1, H, W)
            probs  = torch.sigmoid(logits).squeeze()     # (H, W)
        else:
            output = model(img)
            masks  = output["pred_masks"]                # (1, Q, H', W')
            h, w   = img.shape[-2:]
            masks  = F.interpolate(masks, size=(h, w),
                                   mode="bilinear", align_corners=False)
            probs, _ = torch.sigmoid(masks).squeeze(0).max(dim=0)  # (H, W)

    pred = (probs >= threshold).cpu().numpy().astype(np.uint8) * 255
    return pred


def predict_tta(model, img_tensor: torch.Tensor,
                device: torch.device, model_type: str,
                threshold: float,
                image_size: int = 1024) -> np.ndarray:
    """
    TTA: average predictions over 4 flip augmentations.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    flips = [
        [],
        [A.HorizontalFlip(p=1.0)],
        [A.VerticalFlip(p=1.0)],
        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
    ]

    # Recover numpy image from tensor (un-normalize)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    img_np = np.clip(img_np * std + mean, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    probs_acc = np.zeros((image_size, image_size), dtype=np.float32)

    for flip_list in flips:
        tf   = A.Compose(flip_list + [
            A.Normalize(mean=mean.tolist(), std=std.tolist(), always_apply=True),
            ToTensorV2(),
        ])
        aug  = tf(image=img_np)
        t    = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            if model_type == "segformer":
                logits = model(t)
                p      = torch.sigmoid(logits).squeeze().cpu().numpy()
            else:
                output = model(t)
                masks  = output["pred_masks"]
                h, w   = t.shape[-2:]
                masks  = F.interpolate(masks, size=(h, w),
                                       mode="bilinear", align_corners=False)
                p, _   = torch.sigmoid(masks).squeeze(0).max(dim=0)
                p      = p.cpu().numpy()

        # Un-flip before accumulating
        if any(isinstance(f, A.HorizontalFlip) for f in flip_list):
            p = np.fliplr(p)
        if any(isinstance(f, A.VerticalFlip) for f in flip_list):
            p = np.flipud(p)

        probs_acc += p

    probs_acc /= len(flips)
    return (probs_acc >= threshold).astype(np.uint8) * 255


# ─────────────────────────────────────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_visualization(img_tensor: torch.Tensor,
                       gt_mask: np.ndarray,
                       pred_mask: np.ndarray,
                       out_path: str,
                       metrics: dict) -> None:
    """
    Save a 3-panel visualization: original | GT mask | predicted mask.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = img_tensor.permute(1, 2, 0).numpy()
    img = np.clip(img * std + mean, 0, 1)
    img = (img * 255).astype(np.uint8)

    # GT mask overlay (green)
    gt_overlay = img.copy()
    gt_bin     = (gt_mask > 0)
    gt_overlay[gt_bin] = [0, 255, 0]
    gt_panel   = cv2.addWeighted(img, 0.6, gt_overlay, 0.4, 0)

    # Pred mask overlay (red)
    pred_overlay = img.copy()
    pred_bin     = (pred_mask > 0)
    pred_overlay[pred_bin] = [0, 0, 255]
    pred_panel   = cv2.addWeighted(img, 0.6, pred_overlay, 0.4, 0)

    # Diff overlay: TP=green, FP=blue, FN=red
    diff = img.copy()
    tp   = gt_bin & pred_bin
    fp   = pred_bin & ~gt_bin
    fn   = gt_bin & ~pred_bin
    diff[tp]  = [0, 200, 0]
    diff[fp]  = [200, 0, 0]
    diff[fn]  = [0, 0, 200]

    # Panel labels
    for panel, label in [(gt_panel, "Ground Truth"),
                          (pred_panel, "Prediction"),
                          (diff, f"IoU={metrics.get('iou',0):.3f}")]:
        cv2.putText(panel, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    combined = cv2.hconcat([
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(gt_panel, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(pred_panel, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(diff, cv2.COLOR_RGB2BGR),
    ])
    cv2.imwrite(out_path, combined)


# ─────────────────────────────────────────────────────────────────────────────
#  Report generation
# ─────────────────────────────────────────────────────────────────────────────

def write_report(metrics_per_image: list[dict],
                 aggregate: dict,
                 out_path: str,
                 args,
                 elapsed: float) -> None:
    """Write a structured evaluation_report.txt."""
    lines = [
        "=" * 65,
        "  Aerial Lane Detection — Evaluation Report",
        "=" * 65,
        f"  Checkpoint : {args.checkpoint}",
        f"  Dataset    : {args.dataset} [{args.split}]",
        f"  Threshold  : {args.threshold}",
        f"  TTA        : {args.tta}",
        f"  Samples    : {len(metrics_per_image)}",
        f"  Time       : {elapsed:.1f}s",
        "=" * 65,
        "",
        "── Aggregate Metrics ─────────────────────────────────────────",
        f"  IoU       : {aggregate.get('iou', 0):.4f}",
        f"  Dice      : {aggregate.get('dice', 0):.4f}",
        f"  Precision : {aggregate.get('precision', 0):.4f}",
        f"  Recall    : {aggregate.get('recall', 0):.4f}",
        f"  F1        : {aggregate.get('f1', 0):.4f}",
        "",
        "── Per-Image Results ─────────────────────────────────────────",
    ]

    for i, m in enumerate(metrics_per_image):
        lines.append(
            f"  [{i+1:04d}] "
            f"iou={m.get('iou',0):.4f}  "
            f"dice={m.get('dice',0):.4f}  "
            f"prec={m.get('precision',0):.4f}  "
            f"rec={m.get('recall',0):.4f}  "
            f"f1={m.get('f1',0):.4f}"
        )

    lines += [
        "",
        "=" * 65,
        "  Performance Goal: IoU > 0.85",
        f"  Status: {'✓ ACHIEVED' if aggregate.get('iou', 0) >= 0.85 else '✗ NOT YET (keep training)'}",
        "=" * 65,
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Evaluate] Report saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    print(f"[Evaluate] Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────
    cfg: dict = {
        "model_type":  os.environ.get("MODEL_TYPE", "segformer"),
        "backbone":    "nvidia/mit-b5",
        "num_classes": 2,
        "image_size":  1024,
    }
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))

    if args.model:
        cfg["model_type"] = args.model

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"\n[Evaluate] Loading {args.split} split from {args.dataset}")
    tf = get_val_transforms(cfg["image_size"])

    try:
        dataset = AerialLanesDataset(
            root=args.dataset, split=args.split, transform=tf,
            image_size=cfg["image_size"],
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[Evaluate] ERROR: {e}")
        print("See dataset/README_DOWNLOAD.md")
        sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\n[Evaluate] Building model: {cfg['model_type']}")
    model = build_model(cfg["model_type"], cfg).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[Evaluate] ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Evaluate] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"[Evaluate] Best IoU during training: {ckpt.get('best_iou', 'N/A')}")

    # ── Output dirs ───────────────────────────────────────────────────────
    out_dir     = Path(args.output)
    vis_dir     = out_dir / "eval_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluation loop ───────────────────────────────────────────────────
    tracker          = MetricTracker()
    metrics_per_img  = []
    vis_count        = 0
    t0               = time.time()

    print(f"\n[Evaluate] Running on {len(dataset)} samples (TTA={args.tta})…\n")

    for idx in range(len(dataset)):
        sample     = dataset[idx]
        img_tensor = sample["pixel_values"]    # (C, H, W)
        gt_mask_t  = sample["labels"]          # (H, W) float {0,1}
        gt_mask    = (gt_mask_t.numpy() > 0.5).astype(np.uint8) * 255

        # Inference
        if args.tta:
            pred_mask = predict_tta(model, img_tensor, device,
                                    cfg["model_type"], args.threshold,
                                    cfg["image_size"])
        else:
            pred_mask = predict_single(model, img_tensor, device,
                                       cfg["model_type"], args.threshold)

        # Metrics
        m = compute_metrics(pred_mask, gt_mask, threshold=127)
        metrics_per_img.append(m)
        tracker.update(m)

        if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
            agg = tracker.compute()
            print(f"  [{idx+1}/{len(dataset)}] {format_metrics(agg)}")

        # Save visualizations
        if vis_count < args.vis_count:
            vis_path = str(vis_dir / f"eval_{idx:04d}.jpg")
            save_visualization(img_tensor, gt_mask, pred_mask, vis_path, m)
            vis_count += 1

    elapsed   = time.time() - t0
    aggregate = tracker.compute()

    print(f"\n{'='*55}")
    print(f"  Final Results ({len(dataset)} images, {elapsed:.1f}s)")
    print(f"{'='*55}")
    print(f"  IoU       : {aggregate.get('iou',0):.4f}")
    print(f"  Dice      : {aggregate.get('dice',0):.4f}")
    print(f"  Precision : {aggregate.get('precision',0):.4f}")
    print(f"  Recall    : {aggregate.get('recall',0):.4f}")
    print(f"  F1        : {aggregate.get('f1',0):.4f}")
    print(f"{'='*55}\n")

    # Report
    report_path = str(out_dir / "evaluation_report.txt")
    write_report(metrics_per_img, aggregate, report_path, args, elapsed)

    print(f"[Evaluate] Visualizations ({vis_count}) → {vis_dir}")
    print(f"[Evaluate] Done.")


if __name__ == "__main__":
    main()
