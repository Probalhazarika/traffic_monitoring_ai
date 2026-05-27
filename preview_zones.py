"""
preview_zones.py
────────────────────────────────────────────────────────────────
Draws LANE_ZONES and FREE_TURN_ZONES on the calibration frame
so you can visually verify the bounding box positions.

Usage:
    python preview_zones.py

Output:
    zone_preview_new.jpg
────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from config import VIDEO_PATH, LANE_ZONES, FREE_TURN_ZONES, LANE_PALETTE

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read frame.")
    exit()

h, w = frame.shape[:2]
print(f"✅ Frame size: {w} × {h}")

overlay = frame.copy()

# ── Draw LANE_ZONES (solid fill + thick border) ────────────────
for idx, (name, poly_frac) in enumerate(LANE_ZONES.items()):
    color = LANE_PALETTE[idx % len(LANE_PALETTE)]
    pts = np.array([(int(x * w), int(y * h)) for x, y in poly_frac], dtype=np.int32)

    # Semi-transparent fill
    cv2.fillPoly(overlay, [pts], color)

    # Thick border on the actual frame
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=4)

    # Label at centroid
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    cv2.putText(frame, name, (cx - 60, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(frame, name, (cx - 60, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)

# ── Blend fill ────────────────────────────────────────────────
cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

# ── Draw FREE_TURN_ZONES (dashed outline only) ─────────────────
for name, poly_frac in FREE_TURN_ZONES.items():
    pts = np.array([(int(x * w), int(y * h)) for x, y in poly_frac], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 128, 255), thickness=2)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    cv2.putText(frame, "Free-R", (cx - 50, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)

# ── Draw forward boundary markers ─────────────────────────────
# North stop line  (horizontal): y=0.50, x: 0.425->0.535
cv2.line(frame, (int(0.425*w), int(0.50*h)), (int(0.535*w), int(0.50*h)), (0,0,255), 4)
# South stop line  (horizontal): y=0.90, x: 0.475->0.60
cv2.line(frame, (int(0.475*w), int(0.90*h)), (int(0.60*w), int(0.90*h)),  (0,0,255), 4)
# West stop line   (vertical):   x=0.40, y: 0.65->0.90
cv2.line(frame, (int(0.40*w), int(0.65*h)), (int(0.40*w), int(0.90*h)),   (0,0,255), 4)
# East stop line   (vertical):   x=0.60, y: 0.50->0.70
cv2.line(frame, (int(0.60*w), int(0.50*h)), (int(0.60*w), int(0.70*h)),   (0,0,255), 4)

cv2.putText(frame, "RED = forward stop line  |  Colored box = detection zone  |  Blue outline = Free-R",
            (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

out = "zone_preview_new.jpg"
cv2.imwrite(out, frame)
print(f"📸 Saved → {out}")
