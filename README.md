# AI-Based Smart Traffic Monitoring & Signal Optimization System

**Final Year CSE Major Project** — uses YOLOv8, OpenCV, Flask, SQLite.

---

## 📁 Project Structure

```
traffic_monitoring_ai/
├── config.py               ← All tuneable settings (ROI, thresholds, paths)
├── processor.py            ← Main video pipeline (background thread)
├── app.py                  ← Flask web server + API routes
├── test_detector.py        ← Quick test without Flask
├── requirements.txt
│
├── detector/
│   ├── yolo_detector.py    ← YOLOv8 wrapper (vehicle classes only)
│   ├── roi_manager.py      ← Lane ROI zones + assignment logic
│   └── vehicle_counter.py  ← Count + density classification + annotation
│
├── traffic/
│   └── signal_controller.py ← Green-light timing algorithm (2 strategies)
│
├── database/
│   └── db_manager.py       ← SQLite: insert + query traffic logs
│
├── templates/
│   └── index.html          ← Dashboard UI (Chart.js, live MJPEG, log table)
│
├── static/
│   └── style.css           ← Dark glassmorphism theme
│
└── videos/
    └── traffic.mp4         ← ← ← PUT YOUR VIDEO HERE
```

---

## ⚙️ Setup (Step-by-Step)

### 1. Create & activate a virtual environment
```bash
cd traffic_monitoring_ai
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> YOLOv8 (`yolov8n.pt`) downloads automatically on first run (~6 MB).

### 3. Add your video
Place any traffic footage as:
```
videos/traffic.mp4
```
Free sample videos: https://www.pexels.com/search/videos/traffic/

### 4. (Optional) Adjust ROI for your video
Open `config.py` → edit `LANE_ROIS` to match your video resolution and lane positions.  
Default ROI splits a 1280×720 frame into 4 equal quadrants.  
The ROI is **auto-scaled** to match any video resolution.

---

## 🚀 Running the System

### Quick test (no Flask required)
```bash
python test_detector.py
```
Opens one frame, detects vehicles, prints counts per lane, saves `test_output.jpg`.

### Full dashboard
```bash
python app.py
```
Open browser → **http://localhost:5000**

---

## 📊 Dashboard Features

| Feature | Details |
|---|---|
| Live video feed | MJPEG stream with bounding boxes & ROI overlays |
| Lane cards | Real-time vehicle count + density badge per lane |
| Signal panel | GREEN/RED indicator + green-time bar per lane |
| Bar chart | Vehicle count per lane (auto-refreshes every 2 s) |
| Line chart | Historical trend from DB (last 200 records) |
| Log table | Scrollable recent traffic records with timestamps |

---

## 🔧 Configuration Reference (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `videos/traffic.mp4` | Input video |
| `MODEL_PATH` | `yolov8n.pt` | YOLO model (n/s/m/l/x) |
| `VEHICLE_CLASSES` | `[2,3,5,7]` | car, motorbike, bus, truck |
| `CONFIDENCE_THRESHOLD` | `0.4` | Min detection confidence |
| `DENSITY_LOW_MAX` | `5` | ≤5 vehicles → Low |
| `DENSITY_MEDIUM_MAX` | `15` | ≤15 vehicles → Medium |
| `GREEN_LOW/MEDIUM/HIGH` | `15/30/45 s` | Signal durations |
| `LANE_ROIS` | 4 quadrants | Adjust for your video |

---

## 🧠 How It Works (for Viva)

```
Video Frame
    │
    ▼
YOLODetector        ← YOLOv8 pretrained on COCO
    │ bounding boxes
    ▼
ROIManager          ← bbox centre point falls inside which ROI?
    │ lane → [vehicles]
    ▼
VehicleCounter      ← count per lane → Low / Medium / High
    │ lane_stats
    ▼
SignalController    ← density → green time (15 / 30 / 45 s)
    │ schedule
    ▼
DBManager           ← SQLite insert every 30 frames
    │
    ▼
Flask /api/stats    ← polled by JS every 2 s
    │
    ▼
Dashboard           ← Chart.js charts + MJPEG stream
```

---

## 📝 API Endpoints

| Endpoint | Method | Returns |
|---|---|---|
| `/` | GET | Dashboard HTML |
| `/video_feed` | GET | MJPEG stream |
| `/api/stats` | GET | Current frame lane stats (JSON) |
| `/api/logs?limit=N` | GET | Recent N traffic log records |
| `/api/summary` | GET | Per-lane aggregate statistics |
