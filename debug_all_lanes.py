"""
debug_all_lanes.py — Check detections in ALL lanes across multiple timestamps.
"""
import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import VIDEO_PATH, LANE_ZONES
from detector.yolo_detector import YOLODetector
from detector.lane_detector  import LaneDetector

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25

SAMPLE_SECS = [2, 8, 15, 25, 40]
frames = []
for t in SAMPLE_SECS:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * t))
    ret, f = cap.read()
    if ret: frames.append((t, f))
cap.release()

STREAM_W, STREAM_H = 1280, 720
detector = YOLODetector()
lane_det  = LaneDetector()
lane_det._lane_fracs = dict(LANE_ZONES)
lane_det.calibrated  = True

for (t, frame) in frames[:2]:  # just first 2 for speed
    fh, fw = frame.shape[:2]
    if fw > STREAM_W or fh > STREAM_H:
        frame = cv2.resize(frame, (STREAM_W, STREAM_H))
    h, w = frame.shape[:2]
    lane_det._refresh_polys(w, h)

    all_dets = detector.detect(frame, frame_w=w, frame_h=h)

    print(f"\n{'='*70}")
    print(f" t={t}s | frame {w}×{h} | {len(all_dets)} total detections")
    print(f"{'='*70}")
    print(f"\n{'Zone':8} | px bounds (x1,y1)-(x2,y2)")
    for lname, poly in lane_det.lane_polygons.items():
        pts = poly.reshape(-1,2)
        xs, ys = pts[:,0], pts[:,1]
        print(f"  {lname:6}: x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}")

    for lname, poly in lane_det.lane_polygons.items():
        mask = lane_det._lane_masks.get(lname)
        pts  = poly.reshape(-1,2)
        xs, ys = pts[:,0], pts[:,1]
        x1z,y1z,x2z,y2z = xs.min(),ys.min(),xs.max(),ys.max()

        hits, misses = [], []
        for det in all_dets:
            cx,cy = det["cx"], det["cy"]
            bx1,by1,bx2,by2 = det["bbox"]
            in_poly = cv2.pointPolygonTest(poly,(float(cx),float(cy)),False)
            if mask is not None:
                mbx1=max(0,bx1); mby1=max(0,by1)
                mbx2=min(w,bx2); mby2=min(h,by2)
                ba = max(1,(mbx2-mbx1)*(mby2-mby1))
                oa = int(np.sum(mask[mby1:mby2,mbx1:mbx2]>0))
                ovlp = oa/ba
            else:
                ovlp = 0.0
            entry = (det["label"], det["conf"], cx, cy, in_poly, ovlp)
            if ovlp > 0.05:
                hits.append(entry)
            elif x1z-40<=cx<=x2z+40 and y1z-40<=cy<=y2z+40:
                misses.append(entry)

        print(f"\n  ── {lname} ── in-zone detections (ovlp>5%):")
        if hits:
            for (lb,cf,cx,cy,ip,ov) in hits:
                flag = "✅" if ip>=0 and ov>=0.20 else "⚠️ "
                print(f"    {flag} {lb} {cf:.2f}  cx={cx} cy={cy}  in_poly={ip:+.0f}  ovlp={ov:.2f}")
        else:
            print("    ❌ NONE")

        print(f"  ── {lname} nearby misses:")
        for (lb,cf,cx,cy,ip,ov) in misses[:5]:
            print(f"    ⛔ {lb} {cf:.2f}  cx={cx} cy={cy}  in_poly={ip:+.0f}  ovlp={ov:.2f}")

    # Save annotated image
    debug_img = frame.copy()
    colors = {"North":(0,0,255),"South":(255,0,0),"West":(255,0,255),"East":(0,255,255)}
    for lname, poly in lane_det.lane_polygons.items():
        c = colors.get(lname,(255,255,0))
        cv2.polylines(debug_img,[poly],True,c,3)
        cv2.putText(debug_img,lname,tuple(poly.reshape(-1,2)[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,c,2)
    for det in all_dets:
        x1,y1,x2,y2=det["bbox"]
        cv2.rectangle(debug_img,(x1,y1),(x2,y2),(255,255,255),1)
        cv2.circle(debug_img,(det["cx"],det["cy"]),3,(0,255,0),-1)
    cv2.imwrite(f"debug_all_{t}s.jpg", debug_img)
    print(f"\n📸 Saved → debug_all_{t}s.jpg")
