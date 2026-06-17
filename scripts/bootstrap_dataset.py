#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  bootstrap_dataset.py
#  Generate training dataset from your own drone footage using the existing
#  AutoLaneDetector heuristics as a pseudo-label generator.
#
#  This lets you start training WITHOUT downloading AerialLanes18.
#  Use your own traffic videos → auto-generate lane masks → fine-tune SegFormer.
#
#  Usage
#  ─────
#    python3 bootstrap_dataset.py                              # default video
#    python3 bootstrap_dataset.py --video videos/traffic.mp4
#    python3 bootstrap_dataset.py --video videos/traffic.mp4 --sample-rate 15
#    python3 bootstrap_dataset.py --split 0.8 0.1 0.1         # train/val/test
#
#  Output
#  ──────
#    dataset/
#      train/images/*.jpg   +   train/masks/*.png
#      val/images/*.jpg     +   val/masks/*.png
#      test/images/*.jpg    +   test/masks/*.png
#
#  After running, verify with:
#    python3 train.py --verify-only
#  Then start training:
#    python3 train.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from detector.auto_lane_detector import AutoLaneDetector
from config import VIDEO_PATH


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Bootstrap lane segmentation dataset from drone footage"
    )
    p.add_argument("--video",       default=VIDEO_PATH,
                   help=f"Input video path (default: {VIDEO_PATH})")
    p.add_argument("--output",      default="dataset",
                   help="Output dataset root directory")
    p.add_argument("--sample-rate", type=int, default=10,
                   help="Sample every Nth frame (default: 10)")
    p.add_argument("--max-frames",  type=int, default=500,
                   help="Maximum number of frames to extract (default: 500)")
    p.add_argument("--calib-frames",type=int, default=60,
                   help="Frames used to calibrate AutoLaneDetector (default: 60)")
    p.add_argument("--split",       type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Train/val/test split ratios (default: 0.8 0.1 0.1)")
    p.add_argument("--image-size",  type=int, default=1024,
                   help="Resize frames to this size (default: 1024)")
    p.add_argument("--mask-dilate", type=int, default=8,
                   help="Dilate lane mask by N pixels to thicken lines (default: 8)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--visualize",   action="store_true",
                   help="Save debug overlay images to dataset/debug_vis/")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Lane zone → binary mask
# ─────────────────────────────────────────────────────────────────────────────

def zones_to_mask(zones: dict, frame_w: int, frame_h: int,
                  mask_dilate: int = 8) -> np.ndarray:
    """
    Convert lane zone polygons (fractional coords) to a binary segmentation mask.

    The lane zones define the queue regions (polygons). We:
    1. Fill each polygon → solid lane region mask
    2. Extract the perimeter / edges (Canny) → thin lane boundary mask
    3. Combine both: solid fill + edges
    4. Optionally dilate to give the model thicker targets

    Returns (H, W) uint8 mask, values 0 or 255.
    """
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

    for lane_name, poly_frac in zones.items():
        if not poly_frac or len(poly_frac) < 3:
            continue

        # Convert fractional → pixel coords
        pts = np.array([
            [int(xf * frame_w), int(yf * frame_h)]
            for xf, yf in poly_frac
        ], dtype=np.int32)

        # Fill the polygon (lane queue region = road surface)
        cv2.fillPoly(mask, [pts], 255)

    # Optional: extract just the road edges (boundary of polygon)
    # Then OR with fill to get full road surface mask
    if mask_dilate > 0:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (mask_dilate, mask_dilate))
        mask = cv2.dilate(mask, k, iterations=1)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, sample_rate: int,
                   max_frames: int, calib_frames: int) -> tuple:
    """
    Extract frames from video for calibration and dataset generation.

    Returns
    -------
    (calib_frames_list, sample_frames_list)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Bootstrap] Video: {video_path}")
    print(f"  {w}×{h} @ {native_fps:.1f}fps  ({total} frames total)")

    all_calib   = []
    all_samples = []
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < calib_frames:
            all_calib.append(frame.copy())

        if frame_idx % sample_rate == 0:
            all_samples.append((frame_idx, frame.copy()))

        if len(all_samples) >= max_frames and frame_idx >= calib_frames:
            break

        frame_idx += 1

    cap.release()
    print(f"[Bootstrap] Collected {len(all_calib)} calibration frames, "
          f"{len(all_samples)} sample frames")
    return all_calib, all_samples


# ─────────────────────────────────────────────────────────────────────────────
#  Save image + mask pair
# ─────────────────────────────────────────────────────────────────────────────

def save_pair(frame: np.ndarray, mask: np.ndarray,
              split: str, stem: str, out_root: Path,
              image_size: int) -> None:
    """Resize and save an image + mask pair to the appropriate split directory."""
    img_dir  = out_root / split / "images"
    mask_dir = out_root / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Resize
    frame_r = cv2.resize(frame, (image_size, image_size),
                          interpolation=cv2.INTER_LINEAR)
    mask_r  = cv2.resize(mask,  (image_size, image_size),
                          interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(img_dir  / f"{stem}.jpg"), frame_r,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(mask_dir / f"{stem}.png"), mask_r)


# ─────────────────────────────────────────────────────────────────────────────
#  Debug visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_debug_vis(frame: np.ndarray, mask: np.ndarray,
                   stem: str, debug_dir: Path) -> None:
    """Save a side-by-side debug image: original | mask overlay."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    h, w = frame.shape[:2]
    small_size = min(640, w)

    frame_s = cv2.resize(frame, (small_size, small_size))
    mask_s  = cv2.resize(mask,  (small_size, small_size),
                          interpolation=cv2.INTER_NEAREST)

    # Colour the mask green
    overlay = frame_s.copy()
    overlay[mask_s > 0] = [0, 200, 0]
    blended = cv2.addWeighted(frame_s, 0.55, overlay, 0.45, 0)

    combined = cv2.hconcat([frame_s, blended])
    cv2.imwrite(str(debug_dir / f"{stem}.jpg"), combined,
                [cv2.IMWRITE_JPEG_QUALITY, 85])


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    out_root  = Path(args.output)
    debug_dir = out_root / "debug_vis"

    train_r, val_r, test_r = args.split
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # ── 1. Extract frames ─────────────────────────────────────────────────
    try:
        calib_frames, sample_frames = extract_frames(
            video_path=args.video,
            sample_rate=args.sample_rate,
            max_frames=args.max_frames,
            calib_frames=args.calib_frames,
        )
    except FileNotFoundError as e:
        print(f"[Bootstrap] ERROR: {e}")
        sys.exit(1)

    if not calib_frames:
        print("[Bootstrap] ERROR: Could not read calibration frames from video.")
        sys.exit(1)

    # ── 2. Run AutoLaneDetector to get zone polygons ──────────────────────
    print(f"\n[Bootstrap] Running AutoLaneDetector on "
          f"{len(calib_frames)} calibration frames…")
    ald   = AutoLaneDetector()
    zones = ald.detect(calib_frames, video_path=args.video)

    if not zones:
        print("[Bootstrap] ERROR: AutoLaneDetector found no zones.")
        print("  Try: increase --calib-frames or ensure the video shows a clear")
        print("  intersection with distinct road surfaces.")
        sys.exit(1)

    print(f"[Bootstrap] Detected {len(zones)} zones: {list(zones.keys())}")

    # Sample frame dimensions
    sample_h, sample_w = calib_frames[0].shape[:2]

    # ── 3. Split frame list ───────────────────────────────────────────────
    indices = list(range(len(sample_frames)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * train_r)
    n_val   = int(n_total * val_r)
    # n_test  = n_total - n_train - n_val

    train_idx = set(indices[:n_train])
    val_idx   = set(indices[n_train:n_train + n_val])

    print(f"\n[Bootstrap] Dataset split:")
    print(f"  Train: {n_train}  Val: {n_val}  "
          f"Test: {n_total - n_train - n_val}")

    # ── 4. Generate masks and save pairs ─────────────────────────────────
    print(f"\n[Bootstrap] Generating {n_total} image-mask pairs…")

    counts = {"train": 0, "val": 0, "test": 0}
    vis_count = 0

    for i, (frame_idx, frame) in enumerate(sample_frames):
        # Determine split
        if i in train_idx:
            split = "train"
        elif i in val_idx:
            split = "val"
        else:
            split = "test"

        stem = f"frame_{frame_idx:07d}"

        # Generate lane mask from zones
        mask = zones_to_mask(zones, sample_w, sample_h,
                              mask_dilate=args.mask_dilate)

        # Save image + mask
        save_pair(frame, mask, split, stem, out_root, args.image_size)
        counts[split] += 1

        # Optional debug visualizations (first 20)
        if args.visualize and vis_count < 20:
            save_debug_vis(frame, mask, stem, debug_dir)
            vis_count += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_total}] saved {stem} → {split}")

    # ── 5. Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Bootstrap dataset generated!")
    print(f"  Output: {out_root.resolve()}")
    print(f"  Train: {counts['train']} pairs")
    print(f"  Val:   {counts['val']} pairs")
    print(f"  Test:  {counts['test']} pairs")
    print(f"  Lane zones used: {list(zones.keys())}")
    if args.visualize:
        print(f"  Debug overlays → {debug_dir}/")
    print(f"{'='*55}")
    print(f"\n  Next steps:")
    print(f"  1. Verify: python3 train.py --verify-only")
    print(f"  2. Train:  python3 train.py")
    print(f"  3. Infer:  python3 infer_video.py --input {args.video} --yolo")


if __name__ == "__main__":
    main()
