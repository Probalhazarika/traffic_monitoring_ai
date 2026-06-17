# ─────────────────────────────────────────────────────────────────────────────
#  infer_video.py
#  End-to-end lane detection inference on drone video files.
#
#  Pipeline per frame
#  ──────────────────
#  1.  Read frame  →  resize to model input size
#  2.  Run SegFormer / Mask2Former  →  binary lane mask
#  3.  LaneGraph.extract()  →  skeleton + centerlines + splines
#  4.  LaneTracker.update() →  temporally-smoothed confirmed lanes
#  5.  YOLOLaneFusion.assign()  →  vehicle → lane association (optional)
#  6.  Draw overlays:
#        • semi-transparent lane boundary mask
#        • lane centerlines with IDs
#        • vehicle bboxes coloured by lane
#        • per-lane density HUD
#        • optional heatmap
#  7.  Encode to output video
#  8.  Export lane graph JSON + vehicle-lane CSV
#
#  Usage
#  ─────
#    python infer_video.py --input videos/traffic.mp4
#    python infer_video.py --input videos/traffic.mp4 --yolo  # fuse YOLO
#    python infer_video.py --input videos/traffic.mp4 --frames 100
#    python infer_video.py --input videos/traffic.mp4 --heatmap
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import argparse
import sys
import os
import time
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.augmentations import get_val_transforms
from models              import build_model
from lane_graph          import LaneGraph, LaneData
from lane_tracker        import LaneTracker
from yolo_lane_fusion    import YOLOLaneFusion


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Aerial Lane Detection Video Inference")
    p.add_argument("--input",      required=True, help="Input video path")
    p.add_argument("--checkpoint", default="weights/best_model.pth")
    p.add_argument("--config",     default="configs/segformer_b5.yaml")
    p.add_argument("--output",     default="outputs/lane_detection_output.mp4")
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--frames",     type=int,   default=None,
                   help="Limit frames (None = full video)")
    p.add_argument("--yolo",       action="store_true",
                   help="Run YOLO and fuse vehicle detections with lanes")
    p.add_argument("--heatmap",    action="store_true",
                   help="Overlay traffic density heatmap")
    p.add_argument("--skip",       type=int, default=1,
                   help="Process every Nth frame (1=all, 2=every other…)")
    p.add_argument("--model",      default=None)
    p.add_argument("--no-tracker", action="store_true",
                   help="Disable temporal lane tracker (raw per-frame output)")
    p.add_argument("--export-json",action="store_true",
                   help="Export lane graph JSON for last detected frame")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  Model inference helper
# ─────────────────────────────────────────────────────────────────────────────

def infer_frame(model, frame_rgb: np.ndarray,
                transform, device: torch.device,
                model_type: str, threshold: float,
                image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference on a single RGB frame.

    Returns
    -------
    (binary_mask, prob_map)
      binary_mask : (H_orig, W_orig) uint8 {0, 255}
      prob_map    : (H_orig, W_orig) float32 [0,1] — for heatmap
    """
    orig_h, orig_w = frame_rgb.shape[:2]

    # Albumentations transform
    aug    = transform(image=frame_rgb)
    tensor = aug["image"].unsqueeze(0).to(device)   # (1, C, H, W)

    with torch.no_grad():
        if model_type == "segformer":
            logits = model(tensor)                   # (1, 1, H, W)
            probs  = torch.sigmoid(logits).squeeze() # (H, W)
        else:
            output = model(tensor)
            masks  = output["pred_masks"]            # (1, Q, H', W')
            h, w   = tensor.shape[-2:]
            masks  = F.interpolate(masks, size=(h, w),
                                   mode="bilinear", align_corners=False)
            probs, _ = torch.sigmoid(masks).squeeze(0).max(dim=0)

    prob_np = probs.cpu().float().numpy()

    # Upsample back to original frame resolution if needed
    if prob_np.shape != (orig_h, orig_w):
        prob_np = cv2.resize(prob_np, (orig_w, orig_h),
                             interpolation=cv2.INTER_LINEAR)

    binary = (prob_np >= threshold).astype(np.uint8) * 255
    return binary, prob_np


# ─────────────────────────────────────────────────────────────────────────────
#  Overlay drawing
# ─────────────────────────────────────────────────────────────────────────────

LANE_COLORS = [
    (0,   255,   0),   # green
    (255, 165,   0),   # orange
    (0,   165, 255),   # blue
    (255,   0, 255),   # magenta
    (0,   255, 255),   # cyan
    (255, 255,   0),   # yellow
    (255,  80,  80),   # salmon
    (80,  255,  80),   # lime
]


def draw_mask_overlay(frame: np.ndarray,
                      binary_mask: np.ndarray,
                      alpha: float = 0.35) -> np.ndarray:
    """Overlay semi-transparent lane mask (green tint)."""
    overlay = frame.copy()
    overlay[binary_mask > 0] = [0, 180, 0]
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def draw_lanes_overlay(frame: np.ndarray,
                       lanes: dict[int, LaneData],
                       lg: LaneGraph,
                       alpha: float = 0.4) -> np.ndarray:
    """
    Draw semi-transparent filled lane regions + centerlines + IDs.
    """
    overlay = frame.copy()

    for lid, lane in lanes.items():
        color = LANE_COLORS[(lid - 1) % len(LANE_COLORS)]

        # Fill convex hull of spline as translucent band
        if len(lane.spline_x) > 3:
            sx = np.array([int(x) for x in lane.spline_x])
            sy = np.array([int(y) for y in lane.spline_y])

            # Build a thin polygon around the spline (3px each side)
            pts_up   = np.stack([sx, sy - 4], axis=1)
            pts_down = np.stack([sx, sy + 4], axis=1)
            hull     = np.vstack([pts_up, pts_down[::-1]])
            hull     = hull.reshape(-1, 1, 2).astype(np.int32)
            cv2.fillPoly(overlay, [hull], color=color)

    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    # Draw centerlines on top (sharp)
    frame = lg.draw(frame, lanes, draw_spline=True,
                    draw_id=True, thickness=2)
    return frame


def draw_hud(frame: np.ndarray,
             lanes:       dict[int, LaneData],
             density_map: dict[int, dict],
             frame_idx:   int,
             fps:         float,
             n_lanes:     int) -> np.ndarray:
    """Draw HUD box: frame info + per-lane density."""
    h, w   = frame.shape[:2]
    lines  = [
        f"Frame #{frame_idx}   FPS:{fps:.1f}",
        f"Lanes detected: {n_lanes}",
        "─" * 22,
    ]
    for lid, info in sorted(density_map.items()):
        cnt    = info.get("count", 0)
        dens   = info.get("density", "Low")
        color_tag = "L" if dens == "Low" else "M" if dens == "Medium" else "H"
        lines.append(f"Lane {lid}: {cnt}v [{color_tag}]")

    box_w  = 220
    box_h  = 22 + len(lines) * 18
    x0, y0 = 12, 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (60, 60, 60), 1)

    for i, line in enumerate(lines):
        col = (220, 220, 220) if i < 2 else (160, 220, 160)
        cv2.putText(frame, line, (x0 + 8, y0 + 16 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Heatmap overlay
# ─────────────────────────────────────────────────────────────────────────────

class ProbHeatmap:
    """Accumulate probability maps and display decaying heatmap."""

    def __init__(self, h: int, w: int, decay: float = 0.92):
        self.hmap  = np.zeros((h, w), dtype=np.float32)
        self.decay = decay

    def update(self, prob_map: np.ndarray) -> None:
        self.hmap = self.hmap * self.decay + prob_map * (1 - self.decay)

    def overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        hm_norm = np.clip(self.hmap, 0, 1)
        hm_u8   = (hm_norm * 255).astype(np.uint8)
        hm_col  = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, hm_col, alpha, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    print(f"[Infer] Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────
    cfg: dict = {
        "model_type":  os.environ.get("MODEL_TYPE", "segformer"),
        "backbone":    "nvidia/mit-b5",
        "num_classes": 2,
        "image_size":  1024,
    }
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    if args.model:
        cfg["model_type"] = args.model

    image_size  = cfg["image_size"]
    model_type  = cfg["model_type"]

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"[Infer] Loading model: {model_type}")
    model = build_model(model_type, cfg).to(device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"[Infer] Checkpoint loaded (epoch {ckpt.get('epoch','?')})")
    else:
        print(f"[Infer] WARNING: no checkpoint at {ckpt_path} — using random weights.")

    model.eval()

    # ── Lane graph + tracker ──────────────────────────────────────────────
    lg      = LaneGraph(min_lane_length=40, spline_points=300)
    tracker = LaneTracker(ema_alpha=0.3, max_age=8, min_hits=2) \
              if not args.no_tracker else None

    # ── YOLO fusion (optional) ────────────────────────────────────────────
    yolo_det   = None
    fusion     = None
    if args.yolo:
        try:
            from detector.yolo_detector import YOLODetector
            print("[Infer] Loading YOLO detector…")
            yolo_det = YOLODetector()
            fusion   = YOLOLaneFusion(
                max_dist=120.0,
                output_csv="outputs/vehicle_lane_assignments.csv",
            )
        except Exception as e:
            print(f"[Infer] YOLO load failed: {e} — continuing without YOLO.")

    # ── Transform ─────────────────────────────────────────────────────────
    transform = get_val_transforms(image_size)

    # ── Video I/O ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[Infer] ERROR: cannot open video: {args.input}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames   = args.frames if args.frames else total_frames

    print(f"[Infer] Input : {args.input}  ({orig_w}×{orig_h} @ {native_fps:.1f}fps, "
          f"{total_frames} frames)")

    # Output video
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(
        str(out_path), fourcc, native_fps / max(1, args.skip),
        (orig_w, orig_h)
    )

    # Heatmap
    heatmap = ProbHeatmap(orig_h, orig_w) if args.heatmap else None

    # ── Processing loop ───────────────────────────────────────────────────
    frame_idx       = 0
    processed       = 0
    t_start         = time.time()
    last_lanes: dict[int, LaneData] = {}
    last_json:  dict = {}
    fps_smooth      = native_fps

    print(f"[Infer] Processing up to {max_frames} frames (skip={args.skip})…\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        frame_idx += 1
        if (frame_idx - 1) % args.skip != 0:
            continue

        t_f = time.time()

        # BGR → RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Model inference ───────────────────────────────────────────────
        binary_mask, prob_map = infer_frame(
            model, frame_rgb, transform, device,
            model_type, args.threshold, image_size
        )

        if heatmap:
            heatmap.update(prob_map)

        # ── Lane graph extraction ─────────────────────────────────────────
        raw_lanes = lg.extract(binary_mask)

        if tracker is not None:
            confirmed_lanes = tracker.update(raw_lanes)
        else:
            confirmed_lanes = {lid: ld for lid, ld in raw_lanes.items()}

        if confirmed_lanes:
            last_lanes = confirmed_lanes

        display_lanes = last_lanes

        # ── YOLO detection + fusion ───────────────────────────────────────
        detections    = []
        assignments   = []
        density_map: dict[int, dict] = {}

        if yolo_det and fusion:
            try:
                detections  = yolo_det.detect(frame)
                assignments = fusion.assign(detections, display_lanes, frame_idx)
                density_map = fusion.lane_density(assignments, display_lanes)
            except Exception as e:
                pass  # YOLO failed this frame — skip

        # ── Draw overlays ─────────────────────────────────────────────────
        out_frame = frame.copy()

        # 1. Semi-transparent mask
        out_frame = draw_mask_overlay(out_frame, binary_mask, alpha=0.25)

        # 2. Lane bands + centerlines + IDs
        if display_lanes:
            out_frame = draw_lanes_overlay(out_frame, display_lanes, lg, alpha=0.35)

        # 3. Vehicle bboxes (YOLO + fusion)
        if assignments and fusion:
            out_frame = fusion.draw_assignments(
                out_frame, assignments, detections, display_lanes
            )

        # 4. Heatmap
        if heatmap:
            out_frame = heatmap.overlay(out_frame, alpha=0.40)

        # 5. HUD
        elapsed_f = time.time() - t_f
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(elapsed_f, 1e-6))
        out_frame = draw_hud(
            out_frame, display_lanes, density_map,
            frame_idx, fps_smooth, len(display_lanes)
        )

        writer.write(out_frame)
        processed += 1

        if processed % 30 == 0:
            elapsed = time.time() - t_start
            print(f"  [{frame_idx}/{max_frames}] "
                  f"lanes={len(display_lanes)}  "
                  f"fps≈{fps_smooth:.1f}  "
                  f"elapsed={elapsed:.0f}s")

    cap.release()
    writer.release()
    if fusion:
        fusion.close()

    total_time = time.time() - t_start

    # ── Export lane graph JSON ─────────────────────────────────────────────
    if args.export_json and last_lanes:
        json_path = str(out_path.with_suffix(".lanes.json"))
        LaneGraph.save_json(last_lanes, json_path)

    print(f"\n{'='*55}")
    print(f"  Inference complete")
    print(f"  Frames processed : {processed}")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Avg FPS          : {processed / max(total_time, 1e-6):.1f}")
    print(f"  Output video     : {out_path}")
    if fusion:
        print(f"  CSV              : outputs/vehicle_lane_assignments.csv")
        print(fusion.summary())
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
