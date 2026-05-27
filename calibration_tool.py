"""
calibration_tool.py
────────────────────────────────────────────────────────────────
Run this ONCE to extract the first frame of your video and overlay
a percentage coordinate grid on it.

Usage:
    python calibration_tool.py

Output:
    calibration_frame.jpg  ← open this image to measure lane polygons

Instructions:
    1. Open calibration_frame.jpg
    2. Each grid line shows percentage (0.0 → 1.0) of frame width/height
    3. Click on the corners of each lane you want to define
    4. Write the (x_frac, y_frac) values into config.py → LANE_ZONES
       Example:  "Lane 1": [(0.30, 0.0), (0.50, 0.0), (0.45, 0.5), (0.25, 0.5)]
────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from config import VIDEO_PATH

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video. Check VIDEO_PATH in config.py")
    exit()

# Grab frame at 2 seconds in (more representative than frame 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read frame.")
    exit()

h, w = frame.shape[:2]
print(f"✅ Frame size: {w} × {h}")

# ── Draw percentage grid ───────────────────────────────────────
overlay = frame.copy()
GRID_STEPS = 20   # 0.05 increments

for i in range(1, GRID_STEPS):
    frac = i / GRID_STEPS

    # Vertical line at x = frac*w
    x = int(frac * w)
    cv2.line(overlay, (x, 0), (x, h), (255, 255, 0), 1)
    cv2.putText(overlay, f"{frac:.2f}", (x + 4, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Horizontal line at y = frac*h
    y = int(frac * h)
    cv2.line(overlay, (0, y), (w, y), (255, 255, 0), 1)
    cv2.putText(overlay, f"{frac:.2f}", (8, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

# ── Border labels ──────────────────────────────────────────────
cv2.putText(frame, "X →", (w - 120, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
cv2.putText(frame, "Y ↓", (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
cv2.putText(frame, "0,0", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
cv2.putText(frame, "1,1", (w - 120, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

cv2.putText(frame,
            "Use x_frac, y_frac values to define LANE_ZONES in config.py",
            (w // 2 - 500, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

out = "calibration_frame.jpg"
cv2.imwrite(out, frame)
print(f"\n📸 Saved → {out}")
print(f"   Open this image and note the (x_frac, y_frac) corners of each lane.")
print(f"""
Then edit config.py → LANE_ZONES, for example:

LANE_ZONES = {{
    "Lane 1 (North-In)": [(0.35, 0.0), (0.50, 0.0), (0.50, 0.45), (0.35, 0.45)],
    "Lane 2 (South-In)": [(0.50, 0.55), (0.65, 0.55), (0.65, 1.0), (0.50, 1.0)],
    "Lane 3 (East-In)":  [(0.55, 0.35), (1.0, 0.35), (1.0, 0.50), (0.55, 0.50)],
    "Lane 4 (West-In)":  [(0.0, 0.50), (0.45, 0.50), (0.45, 0.65), (0.0, 0.65)],
}}
""")
