# AerialLanes18 Dataset — Download & Setup Instructions

The **AerialLanes18** dataset was introduced in:

> *"Learning to Detect Lane Markings from Aerial Images"*  
> Bhanu Vinzamuri, Prashanth Reddy Marpu — IGARSS 2018  
> DOI: [10.1109/IGARSS.2018.8519023](https://doi.org/10.1109/IGARSS.2018.8519023)

---

## Step 1 — Download

The dataset is hosted on IEEE DataPort (free academic registration required):

```
https://ieee-dataport.org/documents/aerial-lane-dataset
```

Alternatively, check the official project page / GitHub:
```
https://github.com/MaybeShewill-CV/lanenet-lane-detection
```

> **Note:** If the AerialLanes18 dataset is unavailable, you can use any aerial lane dataset with binary masks. The `AerialLanesDataset` class accepts any image + binary mask pair.

---

## Step 2 — Alternative Public Datasets

If AerialLanes18 is unavailable, these aerial / drone lane datasets work with this pipeline:

| Dataset | Source | Notes |
|---------|--------|-------|
| **TuSimple** | https://github.com/TuSimple/tusimple-benchmark | Road-level, easy |
| **CULane** | https://xingangpan.github.io/projects/CULane.html | Urban, curved |
| **ELAS** | https://github.com/rodrigoberriel/ego-lane-analysis-system | Aerial-ish |
| **Custom drone footage** | your own | Convert frames → masks |

---

## Step 3 — Convert to Required Format

The pipeline expects this directory structure:

```
dataset/
├── train/
│   ├── images/    ← RGB images (.jpg or .png)
│   └── masks/     ← Binary masks (.png, same stem as images)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Mask convention:**
- Pixel value **0** = background (road, vehicles, sky...)
- Pixel value **255** (or 1) = lane marking

---

## Step 4 — Generate Masks from Your Own Drone Footage

If you have your own drone footage without masks, use the existing
`AutoLaneDetector` to bootstrap an initial mask dataset:

```bash
python3 -c "
from detector.auto_lane_detector import AutoLaneDetector
import cv2, numpy as np
from pathlib import Path

cap = cv2.VideoCapture('videos/traffic.mp4')
out_img  = Path('dataset/train/images')
out_mask = Path('dataset/train/masks')
out_img.mkdir(parents=True, exist_ok=True)
out_mask.mkdir(parents=True, exist_ok=True)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if frame_idx % 30 == 0:   # sample every 30th frame
        stem = f'frame_{frame_idx:06d}'
        cv2.imwrite(str(out_img / f'{stem}.jpg'), frame)
        # Placeholder mask — replace with real annotations
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.imwrite(str(out_mask / f'{stem}.png'), mask)
    frame_idx += 1
print(f'Saved {frame_idx//30} frame pairs.')
"
```

Then annotate masks using [LabelMe](https://github.com/wkentaro/labelme) or
[CVAT](https://github.com/opencv/cvat).

---

## Step 5 — Verify Dataset

Run the built-in verifier before training:

```bash
source venv/bin/activate
python3 train.py --verify-only
```

Expected output:
```
[Dataset] ✓ train: 1234 valid pairs (1234 imgs, 1234 masks)
[Dataset] ✓ val  :  246 valid pairs (246 imgs, 246 masks)
[Dataset] ✓ test :  123 valid pairs (123 imgs, 123 masks)
[Dataset] Verification complete.
```

---

## Step 6 — Start Training

```bash
source venv/bin/activate

# SegFormer-B5 (recommended, faster to train)
python3 train.py --config configs/segformer_b5.yaml

# Native Mask2Former (slower, higher quality)
python3 train.py --config configs/mask2former.yaml

# Resume from checkpoint
python3 train.py --resume weights/checkpoint_ep005.pth

# Monitor training
tensorboard --logdir runs/
```

---

## Step 7 — Run Inference on Your Traffic Video

```bash
# Lane detection only
python3 infer_video.py --input videos/traffic.mp4

# Lane detection + YOLO vehicle fusion
python3 infer_video.py --input videos/traffic.mp4 --yolo

# With heatmap + JSON export
python3 infer_video.py --input videos/traffic.mp4 --yolo --heatmap --export-json
```

Output: `outputs/lane_detection_output.mp4`
