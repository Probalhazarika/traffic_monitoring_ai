"""
debug_detection.py
──────────────────
Run this to see RAW YOLO detections (before ROI or class filtering).
It will print every object detected and save an annotated frame.

Usage:
    python debug_detection.py
"""

import cv2
from ultralytics import YOLO

VIDEO_PATH = "videos/traffic.mp4"
MODEL_PATH = "yolov8n.pt"
TEST_FRAME  = 30   # which frame number to test on

# COCO class names for reference
COCO_NAMES = {
    0:"person", 1:"bicycle", 2:"car", 3:"motorcycle",
    4:"airplane", 5:"bus", 6:"train", 7:"truck",
    8:"boat", 9:"traffic light", 10:"fire hydrant",
    15:"cat", 16:"dog", 17:"horse"
}

print(f"\n{'='*55}")
print(f"  DEBUG: Raw YOLO detections on frame #{TEST_FRAME}")
print(f"{'='*55}\n")

# Load model
model = YOLO(MODEL_PATH)

# Grab one frame
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ ERROR: Cannot open video. Check VIDEO_PATH.")
    exit()

cap.set(cv2.CAP_PROP_POS_FRAMES, TEST_FRAME)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ ERROR: Could not read frame. Video may be too short.")
    exit()

h, w = frame.shape[:2]
print(f"✅ Frame size: {w}x{h}\n")

# Run YOLO with low threshold to see everything
results = model(frame, verbose=False, conf=0.1)[0]

if len(results.boxes) == 0:
    print("⚠️  YOLO detected NOTHING in this frame.")
    print("   → Try a different frame or a different video.\n")
else:
    print(f"🔍 YOLO found {len(results.boxes)} objects total:\n")
    print(f"  {'cls_id':<8} {'class_name':<15} {'confidence':<12} {'bbox center (cx,cy)'}")
    print(f"  {'-'*55}")

    vehicle_count = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        name = COCO_NAMES.get(cls_id, f"class_{cls_id}")
        is_vehicle = cls_id in [2, 3, 5, 7]

        tag = " ← VEHICLE ✅" if is_vehicle else ""
        if is_vehicle:
            vehicle_count += 1

        print(f"  {cls_id:<8} {name:<15} {conf:.2f}{'':6} ({cx}, {cy}){tag}")

        # Draw box on frame
        color = (0, 255, 0) if is_vehicle else (100, 100, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {conf:.0%}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    print(f"\n  Total vehicles (car/moto/bus/truck): {vehicle_count}")

    if vehicle_count == 0:
        print("\n⚠️  YOLO sees objects but NONE are vehicles (class 2,3,5,7).")
        print("   Possible causes:")
        print("   → Video angle is too overhead — try yolov8s.pt or yolov8m.pt")
        print("   → Video resolution is very low")
        print("   → Wrong video — no actual vehicles in frame")

# Draw 4-quadrant ROI grid for reference
cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 0), 2)
cv2.line(frame, (0, h//2), (w, h//2), (255, 255, 0), 2)
cv2.putText(frame, "Lane 1", (10,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
cv2.putText(frame, "Lane 2", (w//2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,165,0), 2)
cv2.putText(frame, "Lane 3", (10,  h//2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
cv2.putText(frame, "Lane 4", (w//2+10, h//2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)

# Save output
out_path = "debug_output.jpg"
cv2.imwrite(out_path, frame)
print(f"\n📸 Annotated frame saved → {out_path}")
print("   Green boxes = detected vehicles | Grey boxes = other objects\n")
