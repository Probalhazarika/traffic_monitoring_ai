# ─────────────────────────────────────────────────
#  training/train.py
#
#  Fine-tune YOLOv8s on VisDrone aerial dataset.
#
#  Prerequisites:
#    python training/prepare_visdrone.py
#
#  Usage:
#    python training/train.py [--epochs 50] [--imgsz 1280]
#
#  Output:
#    runs/detect/visdrone_finetune/weights/best.pt
#    → copy to yolov8s-visdrone.pt and update config.py
# ─────────────────────────────────────────────────

import argparse
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent.parent


def train(epochs: int = 50, imgsz: int = 1280, batch: int = 8):
    print("=" * 60)
    print("  VisDrone Fine-Tuning — YOLOv8s")
    print(f"  Epochs: {epochs}  |  imgsz: {imgsz}  |  batch: {batch}")
    print("=" * 60)

    model = YOLO(str(BASE / "yolov8s.pt"))   # start from COCO pretrained

    results = model.train(
        data        = str(BASE / "training" / "visdrone.yaml"),
        epochs      = epochs,
        imgsz       = imgsz,
        batch       = batch,
        name        = "visdrone_finetune",
        patience    = 15,           # early stopping
        save        = True,
        plots       = True,
        # Aerial-specific augmentation
        degrees     = 0.0,          # no rotation (aerial footage is fixed angle)
        fliplr      = 0.5,
        flipud      = 0.0,
        mosaic      = 0.8,
        close_mosaic = 10,
        # Small object optimisation
        multi_scale = True,
        overlap_mask = False,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    dest = BASE / "yolov8s-visdrone.pt"
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, dest)
        print(f"\n✅ Best weights saved to: {dest}")
        print("Update config.py: MODEL_PATH = 'yolov8s-visdrone.pt'")
    else:
        print(f"⚠️  Weights not found at {best_weights}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on VisDrone")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=1280)
    parser.add_argument("--batch",  type=int, default=8)
    args = parser.parse_args()

    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
