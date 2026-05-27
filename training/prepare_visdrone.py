# ─────────────────────────────────────────────────
#  training/prepare_visdrone.py
#
#  Downloads and converts VisDrone2019-DET dataset
#  annotations to YOLO format for fine-tuning.
#
#  Usage:
#    python training/prepare_visdrone.py
#
#  Output:
#    training/visdrone_yolo/
#      images/train/  images/val/
#      labels/train/  labels/val/
# ─────────────────────────────────────────────────

import os, shutil, zipfile, urllib.request
from pathlib import Path

# VisDrone class mapping → COCO-compatible vehicle classes
# VisDrone: 1=pedestrian,2=people,3=bicycle,4=car,5=van,6=truck,
#           7=tricycle,8=awning-tricycle,9=bus,10=motor
VISDRONE_TO_YOLO = {
    4: 2,   # car
    5: 7,   # van → truck
    6: 7,   # truck
    9: 5,   # bus
    10: 3,  # motor → motorcycle
}

BASE      = Path(__file__).parent
OUT_DIR   = BASE / "visdrone_yolo"
SPLITS    = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val":   "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}


def download_and_extract(url: str, dest: Path):
    zip_path = dest / "tmp.zip"
    print(f"Downloading {url} …")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting to {dest} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    zip_path.unlink()


def convert_annotation(ann_path: Path, img_w: int, img_h: int) -> list:
    """Convert a single VisDrone .txt annotation to YOLO format lines."""
    lines = []
    with open(ann_path) as f:
        for row in f:
            parts = row.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            cls_id = int(parts[5])
            if cls_id not in VISDRONE_TO_YOLO or w <= 0 or h <= 0:
                continue
            yolo_cls = VISDRONE_TO_YOLO[cls_id]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def process_split(split_name: str, src_root: Path):
    import cv2
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    ann_dir = src_root / "annotations"
    img_dir = src_root / "images"

    converted = 0
    for ann_file in sorted(ann_dir.glob("*.txt")):
        stem     = ann_file.stem
        img_file = img_dir / f"{stem}.jpg"
        if not img_file.exists():
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]

        yolo_lines = convert_annotation(ann_file, w, h)
        if not yolo_lines:
            continue

        shutil.copy(img_file, img_out / img_file.name)
        with open(lbl_out / f"{stem}.txt", "w") as f:
            f.write("\n".join(yolo_lines))
        converted += 1

    print(f"[{split_name}] Converted {converted} images.")


if __name__ == "__main__":
    for split, url in SPLITS.items():
        dl_dir = BASE / f"visdrone_raw_{split}"
        dl_dir.mkdir(exist_ok=True)
        download_and_extract(url, dl_dir)
        process_split(split, dl_dir / f"VisDrone2019-DET-{split}")

    print(f"\n✅ VisDrone dataset ready at: {OUT_DIR}")
    print("Next step: python training/train.py")
