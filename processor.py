# ─────────────────────────────────────────────────
#  processor.py  —  Research-grade video pipeline.
#
#  Architecture (unchanged from original):
#   ┌─ Render thread ──────────────────────────────┐
#   │  Reads every frame at native FPS             │
#   │  Draws polygons, heatmap, annotations, HUD   │
#   │  Encodes JPEG → shared state                 │
#   └──────────────────────────────────────────────┘
#        ↕  shares: last_detections / lane_dets / perf
#   ┌─ Detector thread ────────────────────────────┐
#   │  Runs multi-pass YOLO + WBF fusion           │
#   │  Computes motion density (optical flow)      │
#   │  Runs lane assignment                        │
#   │  Results written atomically via a lock       │
#   └──────────────────────────────────────────────┘
#
#  New research features wired here:
#    • TrafficHeatmap  — decaying Gaussian heatmap
#    • MotionEstimator — per-lane optical flow
#    • Hybrid density  — passed to VehicleCounter
#    • perf_stats      — latency / fps / det counts
# ─────────────────────────────────────────────────

import cv2
import threading
import time
import numpy as np
from collections import deque

from detector.yolo_detector    import YOLODetector
from detector.lane_detector    import LaneDetector
from detector.vehicle_counter  import VehicleCounter
from detector.heatmap          import TrafficHeatmap
from detector.motion_estimator import MotionEstimator
from traffic.signal_controller import SignalController
from database.db_manager       import DBManager
from config import (VIDEO_PATH, CALIBRATION_FRAMES,
                    HEATMAP_ALPHA)

# ── Shared state (thread-safe via lock) ──────────
state_lock = threading.Lock()
state = {
    "frame":            None,   # latest annotated JPEG bytes
    "signal_schedule":  {},     # full signal schedule dict
    "fps":              0,
    "frame_count":      0,
    "is_running":       False,
    "video_finished":   False,
    "lane_polygons":    {},
    "calibrated":       False,
    # Research features
    "heatmap_enabled":  False,
    "perf_stats":       {       # benchmarking metrics
        "detection_latency_ms":  0.0,
        "avg_detections":        0.0,
        "conf_histogram":        [],   # list of 10 bucket counts
        "fps_history":           [],   # last 30 fps samples
    },
}

DB_SAVE_EVERY_N_FRAMES = 30

# Display resolution
STREAM_W, STREAM_H = 1280, 720

# Rolling window for perf stats
_PERF_WINDOW = 30


class VideoProcessor:
    """
    Orchestrates the full research-grade pipeline for one video.
    """

    def __init__(self, video_path: str = VIDEO_PATH):
        self.video_path  = video_path
        self.detector    = YOLODetector()
        self.lane_det    = LaneDetector()
        self.counter     = VehicleCounter()
        self.heatmap     = TrafficHeatmap(w=STREAM_W, h=STREAM_H)
        self.motion_est  = MotionEstimator()
        self.signal_ctrl = SignalController()
        self.db          = DBManager()
        self._thread     = None
        self._stop_flag  = threading.Event()

        # ── Detection state ──────────────────────
        self._det_lock         = threading.Lock()
        self._det_frame        = None   # display-res frame (720p)
        self._det_native_frame = None   # native 4K frame
        self._det_frame_ready  = threading.Event()
        self._last_detections  = []
        self._last_lane_dets   = {}
        self._last_schedule    = {}
        self._last_motion      = {}

        # ── Perf tracking ────────────────────────
        self._det_count_buf = deque(maxlen=_PERF_WINDOW)  # detections/frame
        self._conf_all      = deque(maxlen=500)            # recent conf scores
        self._fps_buf       = deque(maxlen=_PERF_WINDOW)

    # ── Control ──────────────────────────────────

    def start(self):
        self._stop_flag.clear()
        t1 = threading.Thread(target=self._detector_worker, daemon=True)
        t2 = threading.Thread(target=self._run, daemon=True)
        t1.start()
        t2.start()
        self._thread = t2

    def stop(self):
        self._stop_flag.set()
        self._det_frame_ready.set()

    # ── Detector worker ───────────────────────────

    def _detector_worker(self):
        while not self._stop_flag.is_set():
            self._det_frame_ready.wait(timeout=1.0)
            self._det_frame_ready.clear()

            if self._stop_flag.is_set():
                break

            with self._det_lock:
                frame        = self._det_frame
                native_frame = self._det_native_frame

            if frame is None:
                continue

            fh, fw = frame.shape[:2]

            try:
                self.lane_det._refresh_polys(fw, fh)

                # Optical flow motion estimation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion_scores = self.motion_est.update(gray, fw, fh)

                # Multi-pass YOLO + WBF fusion
                dets = self.detector.detect(frame,
                                            native_frame=native_frame,
                                            frame_w=fw, frame_h=fh)

                lane_dets = self.lane_det.assign_vehicles_to_lanes(
                    dets, frame_w=fw, frame_h=fh
                )
                schedule  = self.signal_ctrl.compute_timings(
                    {ln: {"count": len(v), "density": "Low"}
                     for ln, v in lane_dets.items()}
                )

                # Update perf stats
                self._det_count_buf.append(len(dets))
                for d in dets:
                    self._conf_all.append(d["conf"])

            except Exception as e:
                import traceback
                print(f"[Detector] Error: {e}")
                traceback.print_exc()
                continue

            with self._det_lock:
                self._last_detections = dets
                self._last_lane_dets  = lane_dets
                self._last_schedule   = schedule
                self._last_motion     = motion_scores

    # ── Main render loop ──────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Processor] ERROR: Cannot open video: {self.video_path}")
            with state_lock:
                state["is_running"] = False
            return

        with state_lock:
            state["is_running"]     = True
            state["video_finished"] = False

        # ── Calibration ───────────────────────────
        print(f"[Processor] Calibrating lanes ({CALIBRATION_FRAMES} frames)…")
        calib_frames = []
        for _ in range(CALIBRATION_FRAMES):
            ret, frame = cap.read()
            if ret:
                calib_frames.append(frame)
            else:
                break
        self.lane_det.calibrate(calib_frames)
        with state_lock:
            state["lane_polygons"] = self.lane_det.get_polygons_serializable()
            state["calibrated"]    = True
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / native_fps
        print(f"[Processor] Native FPS: {native_fps:.1f}")

        # ── Loop ──────────────────────────────────
        frame_idx      = 0
        fps_timer      = time.time()
        fps_counter    = 0
        db_counter     = 0

        while not self._stop_flag.is_set():
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.heatmap.reset()
                with state_lock:
                    state["video_finished"] = True
                time.sleep(0.05)
                continue

            with state_lock:
                state["video_finished"] = False

            # Keep native frame for detector zone crops
            native_frame = frame

            # Downscale to display size
            fh, fw = frame.shape[:2]
            if fw > STREAM_W or fh > STREAM_H:
                display_frame = cv2.resize(frame, (STREAM_W, STREAM_H),
                                           interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = frame

            frame = display_frame

            # Post to detector thread (non-blocking)
            with self._det_lock:
                self._det_frame        = display_frame.copy()
                self._det_native_frame = native_frame.copy()
            self._det_frame_ready.set()

            # Grab latest detection results
            with self._det_lock:
                detections   = self._last_detections
                lane_dets    = self._last_lane_dets
                schedule     = self._last_schedule
                motion_scores = self._last_motion

            # ── Draw lane polygons ────────────────
            frame = self.lane_det.draw_lanes(frame)

            # ── Annotate vehicles + hybrid density ─
            frame, lane_stats = self.counter.count_and_annotate(
                frame, lane_dets, motion_scores=motion_scores
            )

            # ── Heatmap overlay ───────────────────
            with state_lock:
                heatmap_on = state["heatmap_enabled"]
            self.heatmap.update(detections)
            if heatmap_on:
                frame = self.heatmap.overlay(frame, alpha=HEATMAP_ALPHA)

            # ── Signal schedule ───────────────────
            if not schedule:
                schedule = self.signal_ctrl.compute_timings(lane_stats)

            # ── HUD overlay ───────────────────────
            frame = self._draw_hud(frame, schedule, frame_idx)

            # ── Encode JPEG ───────────────────────
            _, jpeg = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 82]
            )

            # ── FPS tracking ──────────────────────
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer   = time.time()
                self._fps_buf.append(round(fps, 1))
                with state_lock:
                    state["fps"] = round(fps, 1)

            # ── Update shared state ───────────────
            with state_lock:
                state["frame"]           = jpeg.tobytes()
                state["signal_schedule"] = schedule
                state["frame_count"]     = frame_idx

                # Perf stats for benchmarking panel
                avg_dets = (sum(self._det_count_buf) / len(self._det_count_buf)
                            if self._det_count_buf else 0)
                # Confidence histogram (10 buckets: 0-10%, 10-20%, …, 90-100%)
                conf_hist = [0] * 10
                for c in self._conf_all:
                    bucket = min(9, int(c * 10))
                    conf_hist[bucket] += 1

                state["perf_stats"] = {
                    "detection_latency_ms":  round(self.detector.latency_ms, 1),
                    "avg_detections":        round(avg_dets, 1),
                    "conf_histogram":        conf_hist,
                    "fps_history":           list(self._fps_buf),
                    "sr_enabled":            self.detector.sr_enabled,
                    "wbf_enabled":           True,
                    "fp16_enabled":          self.detector._use_fp16,
                }

            # ── DB logging ────────────────────────
            db_counter += 1
            if db_counter >= DB_SAVE_EVERY_N_FRAMES and lane_stats:
                db_counter = 0
                try:
                    for lane, info in lane_stats.items():
                        self.db.log_traffic(
                            lane          = lane,
                            vehicle_count = info["count"],
                            density       = info["density"],
                            green_time    = (schedule.get(lane, {})
                                            .get("green_time", 15)),
                        )
                except Exception:
                    pass

            frame_idx += 1

            # Real-time pacing — don't run faster than native FPS
            time.sleep(max(0, frame_delay - 0.005))

        cap.release()
        with state_lock:
            state["is_running"] = False

    # ── HUD ───────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray,
                  schedule: dict, frame_idx: int) -> np.ndarray:
        """Draw signal status HUD box in top-right corner."""
        if not schedule:
            return frame

        h, w = frame.shape[:2]
        lines = ["SIGNAL STATUS", f"Frame #{frame_idx}"]
        for lane, info in schedule.items():
            if lane.startswith("__"):
                continue
            count  = info.get("count", 0)
            green  = info.get("green_time", 15)
            sig    = info.get("signal", "RED")
            lines.append(f"{lane}: {count}v. {sig}. Green: {green}s")

        box_w, box_h = 220, 20 + len(lines) * 18
        x0, y0 = w - box_w - 12, 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                      (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h),
                      (60, 60, 60), 1)

        for i, line in enumerate(lines):
            is_header = (i < 2)
            color = (220, 220, 220) if is_header else (180, 180, 180)
            if not is_header:
                sig_word = "GREEN" if "GREEN" in line else "RED"
                color = (80, 220, 80) if sig_word == "GREEN" else (80, 80, 220)
            cv2.putText(frame, line,
                        (x0 + 8, y0 + 16 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, color, 1, cv2.LINE_AA)

        return frame
