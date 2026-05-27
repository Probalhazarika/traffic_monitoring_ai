# ─────────────────────────────────────────────────
#  app.py  –  Flask web dashboard entry point
# ─────────────────────────────────────────────────

import os
import json
from flask import (Flask, render_template, Response,
                   jsonify, request)
from processor import VideoProcessor, state, state_lock
from database.db_manager import DBManager
from config import FLASK_HOST, FLASK_PORT, VIDEO_PATH

app = Flask(__name__)
db  = DBManager()

# ── Start the video processor immediately ────────
processor = VideoProcessor(video_path=VIDEO_PATH)
processor.start()


# ═══════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


# ── MJPEG video stream ───────────────────────────

def _generate_frames():
    """Generator that yields JPEG frames as multipart HTTP stream."""
    import time
    while True:
        with state_lock:
            frame_bytes = state.get("frame")
        if frame_bytes:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + frame_bytes +
                   b"\r\n")
        time.sleep(0.033)   # ~30 fps cap


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── JSON API endpoints ───────────────────────────

@app.route("/api/stats")
def api_stats():
    """Current frame statistics — polled every 2 s by dashboard JS."""
    with state_lock:
        schedule = dict(state["signal_schedule"])
        fps      = state["fps"]
        fc       = state["frame_count"]

    lanes = []
    for lane, info in schedule.items():
        if lane.startswith("__"):
            continue
        lanes.append({
            "lane":       lane,
            "count":      info["count"],
            "density":    info["density"],
            "green_time": info["green_time"],
            "signal":     info["signal"],
        })

    return jsonify({
        "lanes":       lanes,
        "fps":         fps,
        "frame_count": fc,
    })


@app.route("/api/logs")
def api_logs():
    """Recent 100 traffic log records."""
    limit = request.args.get("limit", 100, type=int)
    logs  = db.get_recent_logs(limit=limit)
    return jsonify(logs)


@app.route("/api/summary")
def api_summary():
    """Per-lane aggregate statistics."""
    summary = db.get_lane_summary()
    return jsonify(summary)


@app.route("/api/lane_map")
def api_lane_map():
    """Return detected lane polygon coordinates (for debugging / visualisation)."""
    with state_lock:
        polygons   = dict(state.get("lane_polygons", {}))
        calibrated = state.get("calibrated", False)
    return jsonify({
        "calibrated": calibrated,
        "lane_count": len(polygons),
        "lanes": polygons,
    })


@app.route("/api/recalibrate", methods=["POST"])
def api_recalibrate():
    """Trigger a fresh lane calibration (reads next N frames from the processor)."""
    processor.lane_det.calibrated = False
    # The processor loop will re-run calibration on its next iteration
    # (graceful — it checks the calibrated flag)
    return jsonify({"status": "recalibration requested"})


# ═══════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Traffic Monitoring Dashboard")
    print(f"  Open: http://localhost:{FLASK_PORT}")
    print(f"{'='*50}\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False,
            threaded=True)
