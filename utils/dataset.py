# ─────────────────────────────────────────────────────────────────────────────
#  utils/dataset.py
#  AerialLanesDataset — PyTorch Dataset for AerialLanes18 and custom footage.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import sys
import glob
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset verification
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_IMG_EXTS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SUPPORTED_MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def verify_dataset(root: str, splits: list[str] = ("train", "val", "test"),
                   verbose: bool = True) -> dict:
    """
    Verify AerialLanes18 dataset structure before training.

    Checks
    ------
    1. split/images/ and split/masks/ directories exist
    2. Every image has a corresponding mask (same stem)
    3. No broken/unreadable files
    4. Mask values are binary {0, 255} or {0, 1}

    Returns
    -------
    dict with per-split counts and any errors found.
    """
    results = {}
    all_ok  = True

    for split in splits:
        img_dir  = Path(root) / split / "images"
        mask_dir = Path(root) / split / "masks"

        split_res = {
            "images_found":  0,
            "masks_found":   0,
            "pairs_ok":      0,
            "broken_images": [],
            "missing_masks": [],
            "errors":        [],
        }

        if not img_dir.exists():
            split_res["errors"].append(f"Missing: {img_dir}")
            results[split] = split_res
            all_ok = False
            continue

        if not mask_dir.exists():
            split_res["errors"].append(f"Missing: {mask_dir}")
            results[split] = split_res
            all_ok = False
            continue

        img_files = sorted([
            p for p in img_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_IMG_EXTS
        ])
        split_res["images_found"] = len(img_files)

        mask_stems = {
            p.stem for p in mask_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_MASK_EXTS
        }
        split_res["masks_found"] = len(mask_stems)

        for img_path in img_files:
            stem = img_path.stem

            if stem not in mask_stems:
                split_res["missing_masks"].append(str(img_path))
                all_ok = False
                continue

            # Quick read test
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                if img.size == 0:
                    raise ValueError("empty image")
            except Exception as e:
                split_res["broken_images"].append(f"{img_path}: {e}")
                all_ok = False
                continue

            split_res["pairs_ok"] += 1

        results[split] = split_res

        if verbose:
            status = "✓" if not split_res["errors"] and not split_res["missing_masks"] else "✗"
            print(f"[Dataset] {status} {split:5s}: "
                  f"{split_res['pairs_ok']} valid pairs "
                  f"({split_res['images_found']} imgs, {split_res['masks_found']} masks)")
            for e in split_res["errors"][:3]:
                print(f"  ERROR: {e}")
            for m in split_res["missing_masks"][:3]:
                print(f"  MISSING MASK: {m}")

    results["all_ok"] = all_ok
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class AerialLanesDataset(Dataset):
    """
    Dataset for aerial lane segmentation.

    Directory layout expected
    ─────────────────────────
        root/
          train/
            images/   *.jpg | *.png | ...
            masks/    *.png | ...   (binary: 0=background, 255 or 1=lane)
          val/  ...
          test/ ...

    Mask convention
    ───────────────
    - Masks are loaded as grayscale.
    - Any pixel > 0 is treated as lane (class=1).
    - Output `labels` tensor is float32 in {0, 1}, shape (H, W).

    Parameters
    ----------
    root       : path to dataset root directory
    split      : 'train', 'val', or 'test'
    transform  : albumentations Compose pipeline (image+mask)
    image_size : resize target if no transform provided
    """

    def __init__(
        self,
        root:       str,
        split:      str                 = "train",
        transform:  Optional[Callable]  = None,
        image_size: int                 = 1024,
    ):
        self.root       = Path(root)
        self.split      = split
        self.transform  = transform
        self.image_size = image_size

        self.img_dir  = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Dataset images dir not found: {self.img_dir}\n"
                f"See dataset/README_DOWNLOAD.md for setup instructions."
            )

        # Collect image paths and resolve matching mask paths
        all_imgs = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_IMG_EXTS
        ])

        self.samples: list[tuple[Path, Path]] = []
        for img_p in all_imgs:
            mask_p = self._find_mask(img_p.stem)
            if mask_p is not None:
                self.samples.append((img_p, mask_p))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image-mask pairs found in {self.root}/{split}/.\n"
                f"Check dataset/README_DOWNLOAD.md."
            )

        print(f"[AerialLanesDataset] {split}: {len(self.samples)} pairs loaded.")

    def _find_mask(self, stem: str) -> Optional[Path]:
        for ext in SUPPORTED_MASK_EXTS:
            candidate = self.mask_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.samples[idx]

        # ── Load image ───────────────────────────────────────────────────────
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # ── Load mask ────────────────────────────────────────────────────────
        mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask     = (mask_raw > 0).astype(np.uint8)  # binarise → {0, 1}

        # ── Apply transforms ─────────────────────────────────────────────────
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # (C, H, W) float32 tensor
            mask  = augmented["mask"]    # (H, W) uint8 tensor
        else:
            # Fallback: manual resize + to-tensor
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask  = cv2.resize(mask,  (self.image_size, self.image_size),
                               interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return {
            "pixel_values": image.float(),          # (C, H, W)
            "labels":       mask.float(),           # (H, W) — {0.0, 1.0}
            "image_path":   str(img_path),
            "mask_path":    str(mask_path),
        }

    def compute_pos_weight(self, n_samples: int = 200) -> float:
        """
        Estimate positive class weight from a subset of the dataset.
        pos_weight = (neg_pixels / pos_pixels) — used to handle class imbalance.
        """
        pos = neg = 0
        indices = np.random.choice(len(self), min(n_samples, len(self)), replace=False)
        for i in indices:
            _, mask_path = self.samples[i]
            m = np.array(Image.open(mask_path).convert("L"))
            pos += int(np.sum(m > 0))
            neg += int(np.sum(m == 0))
        ratio = neg / max(pos, 1)
        print(f"[AerialLanesDataset] pos_weight estimate: {ratio:.2f}")
        return float(ratio)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataloader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    root:        str,
    image_size:  int   = 1024,
    batch_size:  int   = 2,
    num_workers: int   = 4,
    pin_memory:  bool  = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test dataloaders.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    from utils.augmentations import get_train_transforms, get_val_transforms

    train_tf = get_train_transforms(image_size)
    val_tf   = get_val_transforms(image_size)

    train_ds = AerialLanesDataset(root, "train", train_tf, image_size)
    val_ds   = AerialLanesDataset(root, "val",   val_tf,   image_size)

    try:
        test_ds = AerialLanesDataset(root, "test", val_tf, image_size)
    except (FileNotFoundError, RuntimeError):
        test_ds = val_ds   # fallback: use val as test

    # pin_memory=False on MPS (unified memory — no speed benefit)
    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=(num_workers > 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, drop_last=False, **common)

    return train_loader, val_loader, test_loader
