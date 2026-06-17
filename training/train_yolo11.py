"""
train_yolo11.py
================
Trains YOLO11s on the VisDrone dataset using the EXACT same hyper-
parameters as the original YOLOv8s run (visdrone_finetune-3) so the
two models can be compared on equal footing.

Usage:
    source venv/bin/activate
    python train_yolo11.py

Results land in:
    runs/detect/yolo11s_visdrone/results.csv
"""

from pathlib import Path
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_YAML = BASE_DIR / "training" / "visdrone.yaml"
RUN_NAME  = "yolo11s_visdrone"

# ── YOLOv8s baseline metrics (from visdrone_finetune-3/results.csv, epoch 47) ──
YOLOV8S_BASELINE = {
    "model":     "yolov8s-visdrone.pt",
    "epochs":    50,
    "mAP50":     0.5764,   # best epoch 47
    "mAP50_95":  0.3494,
    "precision": 0.6819,
    "recall":    0.5328,
}

def print_banner(msg: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {msg}\n{bar}")


def compare(yolo11_metrics: dict):
    """Print a side-by-side comparison table."""
    print_banner("📊 YOLOv8s  vs  YOLO11s — Performance Comparison")

    headers = ["Metric", "YOLOv8s (baseline)", "YOLO11s (new)", "Δ Change"]
    col_w   = [16, 22, 16, 14]

    def row(*cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, col_w))

    print(row(*headers))
    print("  " + "-" * (sum(col_w) + 2 * (len(col_w) - 1)))

    metrics = [
        ("mAP@50",     "mAP50"),
        ("mAP@50-95",  "mAP50_95"),
        ("Precision",  "precision"),
        ("Recall",     "recall"),
    ]

    for label, key in metrics:
        base = YOLOV8S_BASELINE[key]
        new  = yolo11_metrics.get(key, 0.0)
        delta = new - base
        sign  = "+" if delta >= 0 else ""
        print(row(
            label,
            f"{base*100:.1f}%",
            f"{new*100:.1f}%",
            f"{sign}{delta*100:.2f}%",
        ))

    print()
    winner = "YOLO11s" if yolo11_metrics.get("mAP50", 0) > YOLOV8S_BASELINE["mAP50"] \
             else "YOLOv8s"
    print(f"  🏆  Better mAP@50: {winner}")
    print("=" * 60)


def train():
    print_banner("🚀  YOLO11s VisDrone Fine-Tuning")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Model   : yolo11s.pt  (Ultralytics YOLO11 Small)")
    print(f"  Epochs  : 50   (same as YOLOv8s baseline)")
    print(f"  Device  : cpu")
    print()

    # Load YOLO11s (downloads automatically if not cached)
    model = YOLO("yolo11s.pt")

    # Train with identical hyper-parameters to the YOLOv8s run
    results = model.train(
        data        = str(DATA_YAML),
        epochs      = 50,
        imgsz       = 640,
        batch       = 2,
        patience    = 15,
        device      = "cpu",
        workers     = 8,
        pretrained  = True,
        multi_scale = True,
        amp         = True,
        mosaic      = 0.8,
        close_mosaic= 10,
        name        = RUN_NAME,
        project     = str(BASE_DIR / "runs" / "detect"),
        exist_ok    = True,
        verbose     = True,
    )

    # ── Extract best-epoch metrics ──────────────────────────────────────────────
    import csv
    results_csv = BASE_DIR / "runs" / "detect" / RUN_NAME / "results.csv"

    best = {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}
    if results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for row_data in reader:
                try:
                    m50 = float(row_data["metrics/mAP50(B)"].strip())
                    if m50 > best["mAP50"]:
                        best["mAP50"]     = m50
                        best["mAP50_95"]  = float(row_data["metrics/mAP50-95(B)"].strip())
                        best["precision"] = float(row_data["metrics/precision(B)"].strip())
                        best["recall"]    = float(row_data["metrics/recall(B)"].strip())
                except (KeyError, ValueError):
                    pass

    # ── Print comparison ────────────────────────────────────────────────────────
    compare(best)

    print(f"\n  Weights saved to:")
    print(f"    {BASE_DIR}/runs/detect/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    train()
