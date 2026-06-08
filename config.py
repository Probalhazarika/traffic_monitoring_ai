# ─────────────────────────────────────────────
#  config.py  –  Central configuration file
# ─────────────────────────────────────────────

import os

# ── Paths ────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "traffic.mp4")
MODEL_PATH = "yolov8s-visdrone.pt"   # swap to yolov8s-visdrone.pt after fine-tuning
DB_PATH    = os.path.join(BASE_DIR, "database", "traffic_data.db")

# ────────────────────────────────────────────────────────────────
#  ML Lane Detection Pipeline (SegFormer / Mask2Former)
# ────────────────────────────────────────────────────────────────
#
#  Set ML_LANE_ENABLED = True to switch the live system from heuristic
#  AutoLaneDetector to the trained SegFormer-B5 / Mask2Former model.
#
#  Requirements:
#    1. Run: python3 train.py          (produces weights/best_model.pth)
#    2. Set ML_LANE_ENABLED = True below
#    3. Restart: python3 app.py
#
#  When disabled (default), the existing AutoLaneDetector is used.
# ────────────────────────────────────────────────────────────────

# Master switch: True = ML model, False = heuristic AutoLaneDetector
ML_LANE_ENABLED     = False

# Model type: 'segformer' or 'mask2former'
ML_LANE_MODEL_TYPE  = "segformer"

# Path to trained model checkpoint
ML_LANE_CHECKPOINT  = os.path.join(BASE_DIR, "weights", "best_model.pth")

# Config YAML for the trained model
ML_LANE_CONFIG      = os.path.join(BASE_DIR, "configs", "segformer_b5.yaml")

# Inference image size (must match training image_size)
ML_LANE_IMAGE_SIZE  = 1024

# Binary threshold for lane mask (0.0–1.0)
ML_LANE_THRESHOLD   = 0.5

# Temporal smoother: True = use LaneTracker (EMA), False = raw per-frame
ML_LANE_TRACKER     = True

# Minimum skeleton length (pixels) to keep a detected lane
ML_LANE_MIN_LENGTH  = 40

# Lane-to-vehicle assignment max distance (pixels at display resolution)
ML_LANE_MAX_DIST    = 120.0

# ── Research-grade pipeline flags ──────────────────
# FP16 half-precision on MPS/CUDA (roughly 2× faster, negligible accuracy drop)
USE_FP16 = True

# Heatmap overlay opacity (0.0 = invisible, 1.0 = fully opaque)
HEATMAP_ALPHA = 0.45

# Hybrid density estimation weights (must sum to 1.0)
# YOLO count score: normalised vehicle count per lane
# Occupancy score:  fraction of lane polygon covered by bounding boxes
# Motion score:     mean optical flow magnitude inside lane polygon
DENSITY_WEIGHT_YOLO      = 0.5
DENSITY_WEIGHT_OCCUPANCY = 0.3
DENSITY_WEIGHT_MOTION    = 0.2

# ── YOLO vehicle class IDs (COCO dataset) ──────
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE_THRESHOLD = 0.10   # lowered: overhead/distant cars score 0.10–0.15

# ── Traffic density thresholds (vehicles / lane) ─
# Used for the raw YOLO count → density label mapping
DENSITY_LOW_MAX    = 9    # 0–9   → Low
DENSITY_MEDIUM_MAX = 15   # 10–15 → Medium
                          # >15   → High

# ── Signal green-light durations (seconds) ─────
GREEN_LOW    = 15
GREEN_MEDIUM = 30
GREEN_HIGH   = 45
MIN_GREEN    = 10   # absolute minimum green time
MAX_GREEN    = 60   # absolute maximum green time

# ─────────────────────────────────────────────
#  CV-Based Lane Detection Parameters
# ─────────────────────────────────────────────

# Number of frames used to calibrate lane positions at startup
CALIBRATION_FRAMES = 60

# Canny edge detection thresholds
CANNY_LOW  = 50
CANNY_HIGH = 150

# Gaussian blur kernel (must be odd × odd)
BLUR_KERNEL = (5, 5)

# Hough probabilistic line transform parameters
HOUGH_THRESHOLD    = 40    # minimum votes to accept a line
HOUGH_MIN_LINE_LEN = 60    # minimum line length in pixels (at 1920×1080)
HOUGH_MAX_LINE_GAP = 40    # maximum gap to join collinear segments

# Angle filter — keep lines between these degree values (measured from horizontal)
# 0° = horizontal line (lane dividers in a side-view camera)
# 90° = vertical line
# Set LANE_ANGLE_MIN=0, LANE_ANGLE_MAX=180 to keep ALL angles (best for aerial)
LANE_ANGLE_MIN = 0
LANE_ANGLE_MAX = 180

# Maximum pixel distance between line midpoints to be merged into one boundary
LINE_CLUSTER_DIST = 80

# Maximum angle difference (degrees) to be merged into one boundary cluster
LINE_CLUSTER_ANGLE_DIFF = 25

# Maximum number of distinct lane boundaries to detect
MAX_LANES = 6

# Expected number of lanes — used ONLY when Hough detection fails (fallback mode)
EXPECTED_LANES = 4

# ── Road Region-of-Interest Mask ──────────────
# Polygon that covers ONLY the road area.
# Values are fractional: (x_fraction, y_fraction) of frame size.
# The default below covers the full frame — adjust to exclude sky/buildings.
#
# Example for aerial footage where road occupies full frame:
ROAD_MASK_POLY = [
    (0.0,  1.0),   # bottom-left
    (0.0,  0.05),  # top-left  (exclude very top edge)
    (1.0,  0.05),  # top-right
    (1.0,  1.0),   # bottom-right
]

# ══════════════════════════════════════════════════════════════
#  LANE ZONES  ← AUTO-DETECTED (or set manually to override)
# ══════════════════════════════════════════════════════════════
# Zone polygons as fractional (x, y) coordinates.
# x_frac = 0.0 is left edge, 1.0 is right edge.
# y_frac = 0.0 is top edge,  1.0 is bottom edge.
#
# HOW IT WORKS:
#   • Leave LANE_ZONES = {} and the AutoLaneDetector runs on first launch.
#   • Detected zones are saved here automatically (keyed to AUTO_LANE_VIDEO).
#   • Future launches with the same video skip detection entirely.
#   • Switch to a new video → detection reruns and zones update automatically.
#   • Paste coordinates manually here to permanently override auto-detection.
#
# <<AUTO_LANE_START>>
# AUTO-DETECTED for video: traffic.mp4
AUTO_LANE_VIDEO = "traffic.mp4"
LANE_ZONES = {
    "North": [(0.3359, 0.0), (0.5125, 0.0), (0.5125, 0.6458), (0.3359, 0.6458)],
    "East": [(0.6266, 0.2903), (0.9997, 0.2903), (0.9997, 0.6449), (0.6266, 0.6449)],
    "South": [(0.5013, 0.8958), (0.6258, 0.8958), (0.6258, 0.9995), (0.5013, 0.9995)],
    "West": [(0.0, 0.7245), (0.3766, 0.7245), (0.3766, 0.9995), (0.0, 0.9995)],
}
# <<AUTO_LANE_END>>



# ── FREE RIGHT-TURN ZONES (display only, never counted) ─────────
FREE_TURN_ZONES = {
    # Right-turn slip lanes — adjacent to each main arm
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
