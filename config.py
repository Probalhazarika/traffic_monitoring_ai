# ─────────────────────────────────────────────
#  config.py  –  Central configuration file
# ─────────────────────────────────────────────

import os

# ── Paths ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "traffic.mp4")
MODEL_PATH = "yolov8s-visdrone.pt"   # swap to yolov8s-visdrone.pt after fine-tuning
DB_PATH    = os.path.join(BASE_DIR, "database", "traffic_data.db")

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
#  MANUAL LANE ZONES  ← PRIMARY CONFIGURATION
# ══════════════════════════════════════════════════════════════
# Define lane polygons manually using fractional (x, y) coordinates.
# x_frac = 0.0 is left edge, 1.0 is right edge.
# y_frac = 0.0 is top edge,  1.0 is bottom edge.
#
# HOW TO SET THIS UP:
#   1. Run:  python calibration_tool.py
#   2. Open the saved calibration_frame.jpg
#   3. Read off the (x_frac, y_frac) corners of each real lane
#   4. Fill in the dictionary below
#
# If LANE_ZONES is non-empty → used directly (Hough detection skipped).
# If LANE_ZONES = {}          → falls back to automatic Hough detection.
#
# EXAMPLE for a 4-way aerial intersection:
# LANE_ZONES = {
#     "Lane 1 (North)": [(0.35, 0.0),  (0.50, 0.0),  (0.50, 0.45), (0.35, 0.45)],
#     "Lane 2 (South)": [(0.50, 0.55), (0.65, 0.55), (0.65, 1.0),  (0.50, 1.0) ],
#     "Lane 3 (East)":  [(0.55, 0.35), (1.0,  0.35), (1.0,  0.50), (0.55, 0.50)],
#     "Lane 4 (West)":  [(0.0,  0.50), (0.45, 0.50), (0.45, 0.65), (0.0,  0.65)],
# }
#
# ── SIGNAL ZONES: cars waiting to go LEFT or STRAIGHT ──────────
# Only vehicles inside these boxes count toward signal timing.
# Right-turn lane is EXCLUDED from every box.
#
# Measured precisely from calibration_frame.jpg grid overlay.
LANE_ZONES = {
    # ── NORTH arm ───────────────────────────────────────────────
    # Cars travelling SOUTHWARD (downward in frame).
    # Forward boundary (stop line): y=0.50, x: 0.425 → 0.535
    # Backward boundary: top of visible road y=0.00
    "North": [
        (0.425, 0.00), (0.535, 0.00),
        (0.535, 0.50), (0.425, 0.50),
    ],

    # ── SOUTH arm ───────────────────────────────────────────────
    # Cars travelling NORTHWARD (upward in frame).
    # Forward boundary (stop line): y=0.90, x: 0.475 → 0.60
    # Backward boundary: bottom of visible road y=1.00
    "South": [
        (0.475, 0.90), (0.600, 0.90),
        (0.600, 1.00), (0.475, 1.00),
    ],

    # ── WEST arm ────────────────────────────────────────────────
    # Cars travelling EASTWARD (rightward in frame).
    # Forward boundary (stop line): x=0.40, y: 0.65 → 0.90
    # Backward boundary: left edge of visible road x=0.02
    "West": [
        (0.02, 0.65), (0.40, 0.65),
        (0.40, 0.90), (0.02, 0.90),
    ],

    # ── EAST arm ────────────────────────────────────────────────
    # Cars travelling WESTWARD (leftward in frame).
    # Forward boundary (stop line): x=0.60, y: 0.50 → 0.70
    # Backward boundary: right edge of visible road x=0.98
    "East": [
        (0.60, 0.50), (0.98, 0.50),
        (0.98, 0.70), (0.60, 0.70),
    ],
}

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
