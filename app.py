# ─────────────────────────────────────────────────
#  app.py  –  Flask web dashboard entry point
# ─────────────────────────────────────────────────

import os
import json
import secrets
from flask import (Flask, render_template, Response,
                   jsonify, request, redirect, url_for, session)
from processor import VideoProcessor, state, state_lock
from database.db_manager import DBManager
from config import FLASK_HOST, FLASK_PORT, VIDEO_PATH
from auth import auth_bp, init_users_table

app = Flask(__name__)

# ── Secret key (sessions) ────────────────────────
# Use a fixed env-var in production; fall back to a random key (resets on restart)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# ── Register auth blueprint ──────────────────────
app.register_blueprint(auth_bp)
init_users_table()

db = DBManager()

# ── Start the video processor immediately ────────
processor = VideoProcessor(video_path=VIDEO_PATH)
processor.start()


# ═══════════════════════════════════════════════
#  Auth-page routes
# ═══════════════════════════════════════════════

@app.route("/login")
def login_page():
    """Render the login / sign-up page."""
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("auth.html")


@app.route("/logout")
def logout_page():
    session.clear()
    return redirect(url_for("login_page"))


# ═══════════════════════════════════════════════
#  Main dashboard
# ═══════════════════════════════════════════════

@app.route("/")
def index():
    """Main dashboard page — requires login."""
    if not session.get("user_id"):
        return redirect(url_for("login_page"))
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
        time.sleep(0.033)


@app.route("/video_feed")
def video_feed():
    if not session.get("user_id"):
        return "Unauthorized", 401
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── JSON API endpoints ───────────────────────────

@app.route("/api/stats")
def api_stats():
    """Current frame statistics — polled every 2s by dashboard JS."""
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401

    with state_lock:
        schedule = dict(state["signal_schedule"])
        fps      = state["fps"]
        fc       = state["frame_count"]

    lanes = []
    for lane, info in schedule.items():
        if lane.startswith("__"):
            continue
        lanes.append({
            "lane":         lane,
            "count":        info["count"],
            "density":      info["density"],
            "green_time":   info["green_time"],
            "signal":       info["signal"],
        })

    return jsonify({
        "lanes":       lanes,
        "fps":         fps,
        "frame_count": fc,
    })


@app.route("/api/perf")
def api_perf():
    """Research benchmarking metrics."""
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401

    with state_lock:
        perf = dict(state["perf_stats"])
        heatmap_on = state["heatmap_enabled"]
    perf["heatmap_enabled"] = heatmap_on
    return jsonify(perf)


@app.route("/api/toggle_heatmap", methods=["POST"])
def api_toggle_heatmap():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    with state_lock:
        current = state["heatmap_enabled"]
        state["heatmap_enabled"] = not current
        new_state = state["heatmap_enabled"]
    return jsonify({"heatmap_enabled": new_state})


@app.route("/api/toggle_sr", methods=["POST"])
def api_toggle_sr():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    new_state = processor.detector.toggle_sr()
    return jsonify({"sr_enabled": new_state})


@app.route("/api/logs")
def api_logs():
    """Recent 100 traffic log records."""
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    limit = request.args.get("limit", 100, type=int)
    logs  = db.get_recent_logs(limit=limit)
    return jsonify(logs)


@app.route("/api/summary")
def api_summary():
    """Per-lane aggregate statistics."""
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    summary = db.get_lane_summary()
    return jsonify(summary)


@app.route("/api/lane_map")
def api_lane_map():
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
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
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized"}), 401
    processor.lane_det.calibrated = False
    return jsonify({"status": "recalibration requested"})


# ═══════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  AI Traffic Monitoring — Research Edition")
    print(f"  Open: http://localhost:{FLASK_PORT}/login")
    print(f"{'='*50}\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False,
            threaded=True)
