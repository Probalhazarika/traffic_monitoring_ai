# ─────────────────────────────────────────────
#  config.py  –  Central configuration file
# ─────────────────────────────────────────────

import os

# ── Paths ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "traffic.mp4")
MODEL_PATH = "yolov8s-visdrone.pt"
DB_PATH    = os.path.join(BASE_DIR, "database", "traffic_data.db")

# ── Research-grade pipeline flags ──────────────
# FP16 half-precision on MPS/CUDA (faster, negligible accuracy drop)
USE_FP16 = True

# Heatmap overlay opacity (0.0 = invisible, 1.0 = fully opaque)
HEATMAP_ALPHA = 0.45

# Hybrid density estimation weights (must sum to 1.0)
DENSITY_WEIGHT_YOLO      = 0.5
DENSITY_WEIGHT_OCCUPANCY = 0.3
DENSITY_WEIGHT_MOTION    = 0.2

# ── YOLO vehicle class IDs (COCO dataset) ──────
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE_THRESHOLD = 0.10

# ── Traffic density thresholds (vehicles / lane) ─
DENSITY_LOW_MAX    = 9    # 0–9   → Low
DENSITY_MEDIUM_MAX = 15   # 10–15 → Medium
                          # >15   → High

# ── Signal green-light durations (seconds) ─────
GREEN_LOW    = 15
GREEN_MEDIUM = 30
GREEN_HIGH   = 45
MIN_GREEN    = 10
MAX_GREEN    = 60

# ══════════════════════════════════════════════════════════════
#  MANUAL LANE ZONES  (fractional x,y coordinates)
# ══════════════════════════════════════════════════════════════
# x_frac = 0.0 is left edge, 1.0 is right edge.
# y_frac = 0.0 is top edge,  1.0 is bottom edge.
#
# These are the hand-tuned polygons for videos/traffic.mp4.
# To use a different video, update these coordinates manually.
#
LANE_ZONES = {
    "North": [
        (0.3359, 0.0),
        (0.5125, 0.0),
        (0.5125, 0.6458),
        (0.3359, 0.6458),
    ],
    "East": [
        (0.6266, 0.2903),
        (0.9997, 0.2903),
        (0.9997, 0.6449),
        (0.6266, 0.6449),
    ],
    "South": [
        (0.5013, 0.8958),
        (0.6258, 0.8958),
        (0.6258, 0.9995),
        (0.5013, 0.9995),
    ],
    "West": [
        (0.0,    0.7245),
        (0.3766, 0.7245),
        (0.3766, 0.9995),
        (0.0,    0.9995),
    ],
}

# ── FREE RIGHT-TURN ZONES (display only, never counted) ─────
FREE_TURN_ZONES = {
    "North (Free-R)": [
        (0.535, 0.00), (0.575, 0.00),
        (0.575, 0.50), (0.535, 0.50),
    ],
    "South (Free-R)": [
        (0.425, 0.90), (0.475, 0.90),
        (0.475, 1.00), (0.425, 1.00),
    ],
    "West (Free-R)": [
        (0.02, 0.90), (0.40, 0.90),
        (0.40, 0.95), (0.02, 0.95),
    ],
    "East (Free-R)": [
        (0.60, 0.70), (0.98, 0.70),
        (0.98, 0.75), (0.60, 0.75),
    ],
}

# ── Lane polygon overlay colors (BGR for OpenCV) ──
LANE_PALETTE = [
    (0,   255,   0),   # green
    (255, 165,   0),   # orange
    (0,   165, 255),   # blue
    (255,   0, 255),   # magenta
    (0,   255, 255),   # cyan
    (255, 255,   0),   # yellow
]

# ── Flask ───────────────────────────────────────
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5001
FLASK_DEBUG = False

# ── Dashboard refresh interval (ms) ────────────
DASHBOARD_REFRESH_MS = 2000
