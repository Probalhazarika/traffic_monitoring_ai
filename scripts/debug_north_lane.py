"""
debug_north_lane.py
──────────────────────────────────────────────────────────────
Diagnoses why no cars are being detected in the North lane.

Prints ALL YOLO detections on a sample frame, shows their
coordinates, which lane polygon they fall in, and why they
pass or fail the containment tests.

Also saves an annotated image showing every raw YOLO detection
so you can visually compare against the North lane box.

Usage:
    python3 debug_north_lane.py
──────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VIDEO_PATH, LANE_ZONES, CONFIDENCE_THRESHOLD
from detector.yolo_detector import YOLODetector
from detector.lane_detector  import LaneDetector

# ── Grab a representative frame from the video ────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25

# Try several timestamps to catch frames with North lane traffic
SAMPLE_SECS = [2, 5, 10, 15, 20]
frames = []
for t in SAMPLE_SECS:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * t))
    ret, f = cap.read()
    if ret:
        frames.append((t, f))
cap.release()

STREAM_W, STREAM_H = 1280, 720

detector = YOLODetector()
lane_det  = LaneDetector()
lane_det._lane_fracs = dict(LANE_ZONES)
lane_det.calibrated  = True

OVERLAP_THRESHOLD = 0.20

for (t, frame) in frames:
    # Resize to stream size (same as processor.py does)
    fh, fw = frame.shape[:2]
    if fw > STREAM_W or fh > STREAM_H:
        frame = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_LINEAR)

    h, w = frame.shape[:2]
    lane_det._refresh_polys(w, h)

    # ── Run YOLO ──────────────────────────────────────────────
    detections = detector.detect(frame)
    print(f"\n{'='*65}")
    print(f"  t={t}s  |  frame size: {w}×{h}  |  {len(detections)} raw detections")
    print(f"{'='*65}")

    # ── Print North lane polygon in pixels ────────────────────
    north_poly = lane_det.lane_polygons.get("North")
    north_mask = lane_det._lane_masks.get("North")
    if north_poly is not None:
        pts = north_poly.reshape(-1, 2)
        print(f"\nNorth polygon pixels: {pts.tolist()}")
    else:
        print("⚠️  North polygon NOT built!")

    # ── Check every detection ─────────────────────────────────
    print(f"\n{'Label':<12} {'conf':>5}  {'cx':>5} {'cy':>5}  "
          f"{'x1':>5} {'y1':>5} {'x2':>5} {'y2':>5}  "
          f"{'in_N?':>6}  {'ovlp_N':>7}")
    print("-"*75)

    for det in detections:
        cx, cy   = det["cx"], det["cy"]
        x1,y1,x2,y2 = det["bbox"]
        label    = det["label"]
        conf     = det["conf"]

        # Test 1: centre inside North polygon?
        if north_poly is not None:
            in_north = cv2.pointPolygonTest(north_poly, (float(cx), float(cy)), False)
            in_str   = f"{in_north:+.1f}"
        else:
            in_str   = "N/A"

        # Test 2: overlap fraction with North mask
        if north_mask is not None:
            bx1 = max(0, x1); by1 = max(0, y1)
            bx2 = min(w, x2); by2 = min(h, y2)
            box_area     = max(1, (bx2-bx1)*(by2-by1))
            overlap_area = int(np.sum(north_mask[by1:by2, bx1:bx2] > 0))
            ovlp_frac    = overlap_area / box_area
            ovlp_str     = f"{ovlp_frac:.2f}"
        else:
            ovlp_str = "N/A"

        print(f"{label:<12} {conf:>5.2f}  {cx:>5} {cy:>5}  "
              f"{x1:>5} {y1:>5} {x2:>5} {y2:>5}  "
              f"{in_str:>6}  {ovlp_str:>7}")

    # ── Draw annotated debug image ─────────────────────────────
    debug_img = frame.copy()

    # Draw ALL YOLO detections (raw) in white
    for det in detections:
        x1,y1,x2,y2 = det["bbox"]
        cx,cy = det["cx"], det["cy"]
        cv2.rectangle(debug_img, (x1,y1),(x2,y2),(255,255,255),1)
        cv2.circle(debug_img, (cx,cy), 4, (255,255,255), -1)
        cv2.putText(debug_img, f"{det['label']} {det['conf']:.2f}",
                    (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # Draw North lane polygon in bright red
    if north_poly is not None:
        cv2.polylines(debug_img, [north_poly], True, (0,0,255), 3)
        cv2.putText(debug_img, "NORTH ZONE", (int(w*0.425)+5, int(h*0.25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # Draw all other lane polygons
    colors = [(0,255,0),(255,165,0),(255,0,255),(0,255,255)]
    for i, (lname, lpoly) in enumerate(lane_det.lane_polygons.items()):
        if lname == "North" or lpoly is None: continue
        cv2.polylines(debug_img, [lpoly], True, colors[i%len(colors)], 2)

    out = f"debug_north_{t}s.jpg"
    cv2.imwrite(out, debug_img)
    print(f"\n📸 Saved → {out}")

print("\n✅ Done. Open the debug_north_*.jpg files to visually inspect.")
