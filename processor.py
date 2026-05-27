# ─────────────────────────────────────────────────
#  processor.py
#  Main video-processing pipeline.
#
#  Architecture:
#   ┌─ Render thread ──────────────────────────────┐
#   │  Reads every frame at native FPS             │
#   │  Draws polygons, annotations, HUD            │
#   │  Encodes JPEG → shared state                 │
#   └──────────────────────────────────────────────┘
#        ↕  shares: last_detections / last_lane_dets
#   ┌─ Detector thread ────────────────────────────┐
#   │  Runs YOLO (3-pass) on latest frame          │
#   │  Runs lane assignment                        │
#   │  Results written atomically via a lock       │
#   └──────────────────────────────────────────────┘
#
#  The render thread NEVER waits for YOLO.
#  It always uses the most recent detection result,
#  so the video runs at full native speed.
# ─────────────────────────────────────────────────

import cv2
import threading
import time
import numpy as np

from detector.yolo_detector   import YOLODetector
from detector.lane_detector    import LaneDetector
from detector.vehicle_counter  import VehicleCounter
from traffic.signal_controller import SignalController
from database.db_manager       import DBManager
from config import VIDEO_PATH, CALIBRATION_FRAMES

# ── Shared state (thread-safe via lock) ──────────
state_lock = threading.Lock()
state = {
    "frame":           None,   # latest annotated JPEG bytes
    "signal_schedule": {},     # full signal schedule dict
    "fps":             0,
    "frame_count":     0,
    "is_running":      False,
    "video_finished":  False,
    "lane_polygons":   {},     # {lane_name: [[x,y],...]} serialisable
    "calibrated":      False,  # True once lane detection has run
}

# DB save interval
DB_SAVE_EVERY_N_FRAMES = 30

# ── Rendering resolution ──────────────────────────
STREAM_W, STREAM_H = 1280, 720


class VideoProcessor:
    """
    Orchestrates the full pipeline for one video file.

    Two threads run in parallel:
      • _detector_thread  — YOLO inference + lane assignment (slow, ~200-500ms)
      • _run (render loop) — reads frames, draws, encodes, streams (fast, native FPS)

    The render loop never blocks on YOLO; it reuses the last known detections.
    """

    def __init__(self, video_path: str = VIDEO_PATH):
        self.video_path  = video_path
        self.detector    = YOLODetector()
        self.lane_det    = LaneDetector()
        self.counter     = VehicleCounter()
        self.signal_ctrl = SignalController()
        self.db          = DBManager()
        self._thread     = None
        self._stop_flag  = threading.Event()

        # ── Detection state shared between render & detector threads ──
        self._det_lock         = threading.Lock()
        self._det_frame        = None   # latest display-res frame (720p)
        self._det_native_frame = None   # latest native-res frame (4K)
        self._det_frame_ready  = threading.Event()
        self._last_detections  = []
        self._last_lane_dets   = {}
        self._last_schedule    = {}

    # ── Control ──────────────────────────────────

    def start(self):
        """Launch processing in a daemon background thread."""
        if self._thread and self._thread.is_alive():
            print("[Processor] Already running.")
            return
        self._stop_flag.clear()
        self._det_frame_ready.clear()

        # Start the background detector thread first
        det_thread = threading.Thread(
            target=self._detector_worker,
            daemon=True,
            name="DetectorWorker"
        )
        det_thread.start()

        # Start the main render loop
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="VideoProcessor"
        )
        self._thread.start()
        print("[Processor] Started.")

    def stop(self):
        self._stop_flag.set()
        self._det_frame_ready.set()  # unblock detector if waiting
        if self._thread:
            self._thread.join(timeout=5)
        print("[Processor] Stopped.")

    # ── Detector worker (runs in its own thread) ──

    def _detector_worker(self):
        """
        Continuously pops the latest frame, runs YOLO + lane assignment,
        and writes results to shared state.  Runs as fast as YOLO allows;
        the render thread is never slowed down by this.
        """
        while not self._stop_flag.is_set():
            # Wait until a new frame is posted (or stop is requested)
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
                # Warm-up polys before assignment
                self.lane_det._refresh_polys(fw, fh)

                # Pass native 4K frame so zone crops get full-resolution pixels
                dets      = self.detector.detect(frame,
                                                 native_frame=native_frame,
                                                 frame_w=fw, frame_h=fh)
                lane_dets = self.lane_det.assign_vehicles_to_lanes(
                    dets, frame_w=fw, frame_h=fh
                )
                schedule  = self.signal_ctrl.compute_timings(
                    {ln: {"count": len(v), "density": "Low"}
                     for ln, v in lane_dets.items()}
                )
            except Exception as e:
                import traceback
                print(f"[Detector] Error: {e}")
                traceback.print_exc()
                continue

            with self._det_lock:
                self._last_detections = dets
                self._last_lane_dets  = lane_dets
                self._last_schedule   = schedule

    # ── Internal render loop ──────────────────────

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

        # ── Phase 1: Lane Calibration ─────────────
        print(f"[Processor] Reading {CALIBRATION_FRAMES} frames for lane calibration…")
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

        # Native video FPS for real-time pacing
        native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / native_fps
        print(f"[Processor] Video native FPS: {native_fps:.1f} — "
              f"target frame time: {frame_delay*1000:.1f}ms")

        # ── Phase 2: Main render loop ──────────────
        frame_idx       = 0
        fps_timer       = time.time()
        fps_counter     = 0
        last_frame_time = time.time()
        db_counter      = 0

        while not self._stop_flag.is_set():
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                with state_lock:
                    state["video_finished"] = True
                time.sleep(0.05)
                continue

            with state_lock:
                state["video_finished"] = False

            # ── Keep native-res frame for detector zone crops ──
            # This is the original 4K frame before any downscaling.
            # Zone crops run at 4K so tiny South/North cars are visible.
            native_frame = frame  # reference — not copied yet (no mutation before detect)

            # ── Downscale to stream size for display ──────────
            fh, fw = frame.shape[:2]
            if fw > STREAM_W or fh > STREAM_H:
                display_frame = cv2.resize(frame, (STREAM_W, STREAM_H),
                                           interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = frame

            # Use display_frame for rendering from here on
            frame = display_frame

            # ── Post frames to detector (non-blocking) ──
            with self._det_lock:
                self._det_frame        = display_frame.copy()
                self._det_native_frame = native_frame.copy()  # 4K copy for zone crops
            self._det_frame_ready.set()

            # ── Grab latest detection results (non-blocking) ──
            with self._det_lock:
                detections  = self._last_detections
                lane_dets   = self._last_lane_dets
                schedule    = self._last_schedule

            # ── Draw lane polygons ────────────────
            frame = self.lane_det.draw_lanes(frame)

            # ── Annotate vehicles ─────────────────
            frame, lane_stats = self.counter.count_and_annotate(
                frame, lane_dets
            )

            # ── Recompute schedule for display (use cached from detector) ──
            if not schedule:
                schedule = self.signal_ctrl.compute_timings(lane_stats)

            # ── HUD overlay ───────────────────────
            frame = self._draw_hud(frame, schedule, frame_idx)

            # ── Encode JPEG ───────────────────────
            _, jpeg = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 82]
            )

            # ── Update shared state ───────────────
            with state_lock:
                state["frame"]           = jpeg.tobytes()
                state["signal_schedule"] = schedule
                state["frame_count"]     = frame_idx

            # ── Persist to DB ─────────────────────
            db_counter += 1
            if db_counter >= DB_SAVE_EVERY_N_FRAMES:
                self.db.insert_frame_stats(schedule)
                db_counter = 0

            # ── Real-time pacing ──────────────────
            # Sleep only if we finished faster than native FPS.
            # If we're slower (rare), just continue immediately.
            now        = time.time()
            elapsed    = now - last_frame_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)
            last_frame_time = time.time()

            # ── FPS counter ───────────────────────
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                with state_lock:
                    state["fps"] = fps_counter
                fps_counter = 0
                fps_timer   = time.time()

            frame_idx += 1

        cap.release()
        with state_lock:
            state["is_running"] = False
        print("[Processor] Pipeline finished.")

    # ── HUD overlay ──────────────────────────────

    @staticmethod
    def _draw_hud(frame, schedule: dict, frame_idx: int):
        """Draw a heads-up display showing signal status per lane."""
        h, w = frame.shape[:2]
        hud_x = w - 320

        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x - 8, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, "SIGNAL STATUS", (hud_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f"Frame #{frame_idx}", (hud_x, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        y = 100
        for lane, info in schedule.items():
            if lane.startswith("__"):
                continue
            sig_color = (0, 255, 0) if info["signal"] == "GREEN" else (0, 0, 200)
            cv2.circle(frame, (hud_x + 10, y - 5), 8, sig_color, -1)
            cv2.putText(frame,
                        f"{lane}: {info['count']}v  {info['density']}",
                        (hud_x + 28, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
            cv2.putText(frame,
                        f"  Green: {info['green_time']}s",
                        (hud_x + 28, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 255, 150), 1)
            y += 55

        return frame
