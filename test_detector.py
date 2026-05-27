"""
test_detector.py
────────────────
Quick sanity check — runs YOLOv8 on a single frame of your video
and prints what was detected, WITHOUT launching Flask.

Usage:
    python test_detector.py
"""

import cv2
import sys
from detector.yolo_detector  import YOLODetector
from detector.roi_manager     import ROIManager
from detector.vehicle_counter import VehicleCounter
from config import VIDEO_PATH


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video at: {VIDEO_PATH}")
        print("→  Drop your video file into the 'videos/' folder as 'traffic.mp4'")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read first frame.")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Video frame size: {w}×{h}")

    # Detect
    detector = YOLODetector()
    dets = detector.detect(frame)
    print(f"\nTotal vehicles detected: {len(dets)}")
    for d in dets:
        print(f"  {d['label']:12s}  conf={d['conf']:.2f}  "
              f"cx={d['cx']}  cy={d['cy']}")

    # Assign to lanes
    roi = ROIManager(frame_size=(w, h))
    lane_dets = roi.assign_to_lanes(dets)
    print("\nPer-lane breakdown:")
    for lane, ld in lane_dets.items():
        print(f"  {lane}: {len(ld)} vehicle(s)")

    # Classify density
    counter = VehicleCounter()
    frame, lane_stats = counter.count_and_annotate(frame, lane_dets)
    print("\nDensity classification:")
    for lane, info in lane_stats.items():
        print(f"  {lane}: count={info['count']}  density={info['density']}")

    # Save annotated frame
    out = "test_output.jpg"
    cv2.imwrite(out, frame)
    print(f"\nAnnotated frame saved → {out}")
    print("Open it to verify bounding boxes and ROI overlays look correct.")


if __name__ == "__main__":
    main()
