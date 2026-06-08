# ─────────────────────────────────────────────────────────────────────────────
#  train.py
#  Production training script for aerial lane detection.
#  Supports SegFormer-B5 and native Mask2Former.
#
#  Usage
#  ─────
#    python train.py                              # train with default config
#    python train.py --config configs/mask2former.yaml
#    python train.py --resume weights/checkpoint_ep5.pth
#    python train.py --verify-only               # dataset check only
#    python train.py --epochs 1 --debug          # smoke test (1 batch)
#
#  MODEL_TYPE env var or YAML key selects model:
#    MODEL_TYPE=segformer python train.py
#    MODEL_TYPE=mask2former python train.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import time
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from utils.dataset  import verify_dataset, build_dataloaders
from utils.losses   import CombinedLoss
from utils.metrics  import MetricTracker, format_metrics
from models         import build_model


# ─────────────────────────────────────────────────────────────────────────────
#  Default configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model_type":             os.environ.get("MODEL_TYPE", "segformer"),
    "backbone":               "nvidia/mit-b5",
    "num_classes":            2,
    "image_size":             1024,
    "batch_size":             2,
    "grad_accumulation":      8,
    "lr":                     1e-4,
    "weight_decay":           1e-2,
    "epochs":                 50,
    "early_stopping_patience":10,
    "mixed_precision":        True,
    "loss_dice_weight":       0.5,
    "loss_bce_weight":        0.5,
    "warmup_epochs":          2,
    "scheduler":              "cosine",
    "tensorboard_dir":        "runs/segformer_b5",
    "checkpoint_dir":         "weights",
    "best_model_name":        "best_model.pth",
    "dataset_root":           "dataset",
    "num_workers":            0,
    "threshold":              0.5,
    "log_every_n_steps":      10,
    "mean":                   [0.485, 0.456, 0.406],
    "std":                    [0.229, 0.224, 0.225],
}


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Aerial Lane Detection Training")
    p.add_argument("--config",      default="configs/segformer_b5.yaml",
                   help="YAML config file path")
    p.add_argument("--resume",      default=None,
                   help="Checkpoint path to resume training from")
    p.add_argument("--verify-only", action="store_true",
                   help="Only verify dataset then exit")
    p.add_argument("--debug",       action="store_true",
                   help="Smoke test: train 1 batch per epoch")
    p.add_argument("--epochs",      type=int, default=None,
                   help="Override number of epochs")
    p.add_argument("--batch-size",  type=int, default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--dataset",     default=None, help="Override dataset root")
    p.add_argument("--model",       default=None,
                   help="Model type: segformer or mask2former")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(args) -> dict:
    cfg = DEFAULT_CONFIG.copy()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f)
        cfg.update(yaml_cfg)
        print(f"[Config] Loaded: {config_path}")
    else:
        print(f"[Config] Not found: {config_path} — using defaults.")

    # CLI overrides
    if args.epochs is not None:    cfg["epochs"]       = args.epochs
    if args.batch_size is not None: cfg["batch_size"]  = args.batch_size
    if args.lr is not None:         cfg["lr"]          = args.lr
    if args.dataset is not None:    cfg["dataset_root"] = args.dataset
    if args.model is not None:      cfg["model_type"]  = args.model

    # Env var overrides
    env_model = os.environ.get("MODEL_TYPE")
    if env_model:  cfg["model_type"] = env_model

    print(f"[Config] model_type={cfg['model_type']}  "
          f"backbone={cfg['backbone']}  "
          f"image_size={cfg['image_size']}  "
          f"batch={cfg['batch_size']}  "
          f"grad_accum={cfg['grad_accumulation']}  "
          f"lr={cfg['lr']}  "
          f"epochs={cfg['epochs']}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Device] Apple MPS (Metal GPU) ✅")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU (no GPU detected)")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
#  Autocast context manager (device-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def get_autocast(device: torch.device, enabled: bool = True):
    """Return the appropriate autocast context for the device."""
    if not enabled:
        import contextlib
        return contextlib.nullcontext()
    device_type = device.type
    if device_type == "cuda":
        return torch.amp.autocast("cuda")
    elif device_type == "mps":
        return torch.amp.autocast("cpu")  # MPS uses CPU autocast path
    else:
        import contextlib
        return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
#  Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model, loader, optimizer, loss_fn, scaler,
    device, cfg, writer, epoch, debug=False
) -> dict:
    model.train()
    tracker   = MetricTracker()
    grad_accum = cfg["grad_accumulation"]
    threshold  = cfg["threshold"]
    amp_ctx    = get_autocast(device, cfg["mixed_precision"])
    model_type = cfg["model_type"]

    optimizer.zero_grad()
    total_loss  = 0.0
    step        = 0

    for batch_idx, batch in enumerate(loader):
        imgs   = batch["pixel_values"].to(device)   # (B, C, H, W)
        masks  = batch["labels"].to(device)          # (B, H, W) float

        with amp_ctx:
            if model_type == "segformer":
                logits = model(imgs)                 # (B, 1, H, W)
                loss   = loss_fn(logits, masks)

            elif model_type == "mask2former":
                # Mask2Former expects (B, N, H, W) gt_masks
                # For binary: N=1, label=1 (lane)
                B, H, W = masks.shape
                gt_masks  = masks.unsqueeze(1).float()           # (B, 1, H, W)
                gt_labels = torch.ones(B, 1, dtype=torch.long, device=device)
                output = model(imgs)
                loss   = model.compute_loss(output, gt_masks, gt_labels)

        # Scale loss for gradient accumulation
        loss_scaled = loss / grad_accum
        loss_scaled.backward()
        total_loss += loss.item()

        # Gradient step every grad_accum steps
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % cfg["log_every_n_steps"] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                global_step = epoch * len(loader) + batch_idx
                writer.add_scalar("train/loss", avg_loss, global_step)
                print(f"  [E{epoch}] step={step} loss={avg_loss:.4f}")

        # Compute metrics on predictions
        with torch.no_grad():
            if model_type == "segformer":
                probs = torch.sigmoid(logits.squeeze(1))
            else:
                probs, _ = torch.sigmoid(output["pred_masks"]).max(dim=1)
                import torch.nn.functional as F
                probs = F.interpolate(probs.unsqueeze(1),
                                      size=masks.shape[-2:],
                                      mode="bilinear",
                                      align_corners=False).squeeze(1)

            pred_bin = (probs >= threshold).float()
            gt_bin   = (masks >= 0.5).float()
            tp = (pred_bin * gt_bin).sum()
            fp = (pred_bin * (1 - gt_bin)).sum()
            fn = ((1 - pred_bin) * gt_bin).sum()
            smooth = 1e-6
            iou  = (tp + smooth) / (tp + fp + fn + smooth)
            dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
            tracker.update({"iou": iou.item(), "dice": dice.item(),
                             "loss": loss.item()}, n=imgs.size(0))

        if debug and batch_idx >= 0:
            print("  [Debug] single batch smoke test passed.")
            break

    return tracker.compute()


# ─────────────────────────────────────────────────────────────────────────────
#  Validation epoch
# ─────────────────────────────────────────────────────────────────────────────

def val_epoch(model, loader, loss_fn, device, cfg, debug=False) -> dict:
    model.eval()
    tracker   = MetricTracker()
    threshold = cfg["threshold"]
    model_type = cfg["model_type"]
    amp_ctx   = get_autocast(device, cfg["mixed_precision"])

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            imgs  = batch["pixel_values"].to(device)
            masks = batch["labels"].to(device)

            with amp_ctx:
                if model_type == "segformer":
                    logits = model(imgs)
                    loss   = loss_fn(logits, masks)
                    probs  = torch.sigmoid(logits.squeeze(1))
                else:
                    B, H, W   = masks.shape
                    gt_masks  = masks.unsqueeze(1).float()
                    gt_labels = torch.ones(B, 1, dtype=torch.long, device=device)
                    output    = model(imgs)
                    loss      = model.compute_loss(output, gt_masks, gt_labels)
                    import torch.nn.functional as F
                    probs, _  = torch.sigmoid(output["pred_masks"]).max(dim=1)
                    probs     = F.interpolate(probs.unsqueeze(1),
                                              size=masks.shape[-2:],
                                              mode="bilinear",
                                              align_corners=False).squeeze(1)

            pred_bin = (probs >= threshold).float()
            gt_bin   = (masks >= 0.5).float()
            smooth = 1e-6
            tp = (pred_bin * gt_bin).sum()
            fp = (pred_bin * (1 - gt_bin)).sum()
            fn = ((1 - pred_bin) * gt_bin).sum()
            tn = ((1 - pred_bin) * (1 - gt_bin)).sum()
            iou       = (tp + smooth) / (tp + fp + fn + smooth)
            dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
            precision = (tp + smooth) / (tp + fp + smooth)
            recall    = (tp + smooth) / (tp + fn + smooth)
            f1        = 2 * precision * recall / (precision + recall + smooth)

            tracker.update({
                "iou": iou.item(), "dice": dice.item(),
                "precision": precision.item(), "recall": recall.item(),
                "f1": f1.item(), "loss": loss.item(),
            }, n=imgs.size(0))

            if debug and batch_idx >= 0:
                break

    return tracker.compute()


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"[Checkpoint] Resumed from epoch {ckpt.get('epoch', '?')}: {path}")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args)

    # ── Dataset verification ────────────────────────────────────────────────
    print("\n[Dataset] Verifying dataset structure…")
    vresults = verify_dataset(cfg["dataset_root"],
                               splits=["train", "val", "test"],
                               verbose=True)
    if args.verify_only:
        print("[Dataset] Verification complete.")
        sys.exit(0 if vresults.get("all_ok") else 1)

    if not vresults.get("all_ok"):
        print("\n[Dataset] WARNING: Dataset has issues — continuing anyway.")
        print("See dataset/README_DOWNLOAD.md for setup instructions.\n")

    # ── Device ─────────────────────────────────────────────────────────────
    device = get_device()

    # ── Dataloaders ─────────────────────────────────────────────────────────
    print("\n[Data] Building dataloaders…")
    try:
        train_loader, val_loader, _ = build_dataloaders(
            root=cfg["dataset_root"],
            image_size=cfg["image_size"],
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[Data] ERROR: {e}")
        print("Download the dataset first — see dataset/README_DOWNLOAD.md")
        sys.exit(1)

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\n[Model] Building {cfg['model_type']}…")
    model = build_model(cfg["model_type"], cfg).to(device)
    print(f"[Model] Parameters: {model.num_parameters():,}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = CombinedLoss(
        dice_weight=cfg["loss_dice_weight"],
        bce_weight=cfg["loss_bce_weight"],
    ).to(device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Mask2Former uses different LRs for backbone vs head
    if cfg["model_type"] == "mask2former":
        backbone_params = list(model.backbone.parameters())
        head_params     = [p for n, p in model.named_parameters()
                           if not any(n.startswith("backbone") for _ in [1])]
        optimizer = AdamW([
            {"params": backbone_params, "lr": cfg.get("lr_backbone", 1e-5)},
            {"params": head_params,     "lr": cfg.get("lr_head",     cfg["lr"])},
        ], weight_decay=cfg["weight_decay"])
    else:
        optimizer = AdamW(model.parameters(),
                          lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"])

    # ── Scheduler: warmup + cosine ────────────────────────────────────────────
    warmup_steps  = cfg["warmup_epochs"] * len(train_loader)
    total_steps   = cfg["epochs"] * len(train_loader)
    warmup_sched  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                              total_iters=warmup_steps)
    cosine_sched  = CosineAnnealingLR(optimizer,
                                       T_max=total_steps - warmup_steps,
                                       eta_min=1e-6)
    scheduler     = SequentialLR(optimizer,
                                  schedulers=[warmup_sched, cosine_sched],
                                  milestones=[warmup_steps])

    # GradScaler — only for CUDA (MPS doesn't support float16 in all ops)
    use_scaler = torch.cuda.is_available() and cfg["mixed_precision"]
    scaler     = torch.amp.GradScaler("cuda") if use_scaler else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_iou      = 0.0
    patience_cnt  = 0

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_iou     = ckpt.get("best_iou", 0.0)

    # ── TensorBoard ────────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg["tensorboard_dir"])
    print(f"[TensorBoard] Logs → {cfg['tensorboard_dir']}")
    print(f"  Run: tensorboard --logdir {cfg['tensorboard_dir']}\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"Starting training: epochs {start_epoch}–{cfg['epochs']-1}")
    print(f"{'='*60}")

    ckpt_dir  = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / cfg["best_model_name"]

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        print(f"\n── Epoch {epoch}/{cfg['epochs']-1} ─────────────")

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, cfg, writer, epoch, debug=args.debug
        )
        scheduler.step()

        val_metrics = val_epoch(
            model, val_loader, loss_fn, device, cfg, debug=args.debug
        )

        elapsed = time.time() - t0
        print(f"  Train: {format_metrics(train_metrics, 'train')}")
        print(f"  Val:   {format_metrics(val_metrics,   'val')}   [{elapsed:.0f}s]")

        # TensorBoard
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Checkpoint every epoch
        save_checkpoint({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_metrics":     val_metrics,
            "best_iou":        best_iou,
            "config":          cfg,
        }, str(ckpt_dir / f"checkpoint_ep{epoch:03d}.pth"))

        # Best model
        val_iou = val_metrics.get("iou", 0.0)
        if val_iou > best_iou:
            best_iou    = val_iou
            patience_cnt = 0
            save_checkpoint({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "best_iou":    best_iou,
                "config":      cfg,
            }, str(best_path))
            print(f"  ✓ New best IoU: {best_iou:.4f} — saved to {best_path}")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{cfg['early_stopping_patience']})")

        # Early stopping
        if patience_cnt >= cfg["early_stopping_patience"]:
            print(f"\n[EarlyStopping] No improvement for "
                  f"{cfg['early_stopping_patience']} epochs — stopping.")
            break

        if args.debug:
            print("[Debug] Smoke test complete — exiting after 1 epoch.")
            break

    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete. Best IoU: {best_iou:.4f}")
    print(f"Best model saved to: {best_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
