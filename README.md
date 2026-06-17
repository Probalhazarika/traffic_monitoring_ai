# TrafficCommand: AI-Powered Smart Traffic Monitoring System 🚦

**TrafficCommand** is a production-grade, research-oriented AI traffic monitoring and optimization system. It combines state-of-the-art computer vision models (YOLOv8) with advanced image processing techniques to monitor traffic density, optimize traffic light signals in real-time, and manage a team of traffic controllers via a secure administrative portal.

---

## 🚀 Key Features

### Advanced AI Vision Pipeline
- **YOLOv8 Object Detection**: High-speed, real-time detection of vehicles (cars, trucks, buses, motorbikes).
- **Weighted Box Fusion (WBF)**: Intelligently merges overlapping bounding boxes to drastically reduce false positives.
- **Optical Flow Tracking**: Tracks the vector movement of vehicles across frames to ensure stable counts.
- **Real-ESRGAN Super-Resolution (Super-Res)**: AI upscaling to enhance blurry or distant drone footage before detection.
- **Heatmap Generation**: Visualizes traffic density and accumulation zones dynamically over the video feed.

### Secure Controller Portal & Admin Dashboard
- **Role-Based Access Control**: Separate privileges for standard Traffic Controllers and System Administrators.
- **Admin Panel**: A centralized dashboard to approve, revoke, or delete access for new controller sign-ups.
- **Automated SQLite Database**: Secure credential storage using `werkzeug.security` password hashing.

### Real-Time Analytics Dashboard
- **Live MJPEG Video Feed**: Low-latency video streaming directly to the browser.
- **Dynamic Charting**: Real-time rendering of lane densities and historical traffic trends using Chart.js.
- **Smart Signal Control**: Automatically allocates Green/Red light timing based on real-time lane density classifications (Low/Medium/High).
- **Modern UI**: Built with a sleek, responsive Cyberpunk/Glassmorphism design aesthetic.

---

## 🛠️ Tech Stack

**Core Logic & Computer Vision**
- **Python 3.10+**: The core programming language.
- **Ultralytics YOLOv8**: For primary vehicle detection.
- **OpenCV (`cv2`)**: For video frame manipulation, drawing bounding boxes, and Optical Flow calculations.
- **Real-ESRGAN**: For super-resolution image enhancement.
- **Ensemble Boxes (WBF)**: For Weighted Box Fusion algorithm.

**Backend & Web Server**
- **Flask**: Lightweight WSGI web application framework serving the API and HTML templates.
- **SQLite3**: Serverless database for persisting user credentials, roles, and traffic history logs.
- **Werkzeug**: For secure password hashing and session management.

**Frontend & UI**
- **HTML5 & CSS3**: Custom-built Glassmorphism design system (no bulky CSS frameworks).
- **Vanilla JavaScript**: For asynchronous API polling, DOM manipulation, and interactive features without the overhead of React/Vue.
- **Chart.js**: For rendering the live Bar and Line charts on the dashboard.
- **Google Fonts**: Utilizing *Inter* and *JetBrains Mono* for crisp, modern typography.

---

## ⚙️ Setup & Installation

Follow these steps to get the system running on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Probalhazarika/traffic_monitoring_ai.git
cd traffic_monitoring_ai
```

### 2. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
```bash
python3 -m venv venv
source venv/bin/activate        # On macOS/Linux
# venv\Scripts\activate         # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: YOLOv8 model weights `yolov8n.pt` will automatically download on the first run).*

### 4. Provide Video Footage
Place the traffic video you wish to analyze in the `videos` directory. By default, the system looks for:
```text
videos/traffic.mp4
```

---

## 🚦 How to Run the System

### 1. Start the Flask Server
```bash
python app.py
```
You should see output indicating that the database is initialized and the server is running on `http://127.0.0.1:5001`.

### 2. Access the Application
Open your web browser and navigate to:
**http://localhost:5001**

### 3. Authentication Flow
Because the system is secure, you cannot view the dashboard without an approved account.
1. Go to **http://localhost:5001/signup** and create a new account.
2. *Note: The very first time the database initializes, a default Admin account is automatically generated.*
   - **Username:** `ram`
   - **Password:** `hazarika1?`
3. Go to **http://localhost:5001/login**, check the **"Login as Admin"** box, and log in with the `ram` credentials.
4. From the **Admin Panel**, you can approve the new account you just created.
5. Log out, and log back in with your new approved account to access the main Traffic Dashboard.

---

## 🧠 System Architecture

The pipeline runs efficiently using a multi-threaded approach:

1. **Background Processing Thread (`processor.py`)**
   - Continuously reads frames from `videos/traffic.mp4`.
   - Optionally applies **Super-Resolution** to the frame.
   - Runs **YOLOv8** inference.
   - Applies **Weighted Box Fusion (WBF)** to clean up bounding boxes.
   - Calculates **Optical Flow** to maintain tracking consistency.
   - Assigns vehicles to specific geometric **Lane ROIs** (Regions of Interest).
   - Generates the dynamic **Heatmap**.
   - Evaluates density and calculates signal timings.
   - Writes periodic logs to the SQLite database.

2. **Web Server Thread (`app.py` & `auth.py`)**
   - Manages HTTP routes and API endpoints (`/api/stats`, `/api/perf`, `/api/auth/...`).
   - Serves the processed frames as an MJPEG stream to `/video_feed`.
   - Handles user sessions, authentication checks, and database reads for the Admin Panel.

3. **Client-Side Polling (`index.html`)**
   - The frontend JavaScript securely polls the `/api/stats` and `/api/perf` endpoints every 2 seconds.
   - Updates the DOM and Chart.js graphs instantly without requiring full page reloads.

---

## 🎯 Model Training (VisDrone Dataset)

To ensure the highest accuracy for top-down, oblique, and long-range aerial perspectives, the underlying YOLOv8 model was custom-trained on the **VisDrone Dataset**. 
- **The Dataset**: VisDrone is a large-scale benchmark dataset explicitly designed for drone-based computer vision. It contains thousands of images and videos captured by various drone-mounted cameras across different urban and highway scenarios.
- **Why VisDrone?**: Standard pre-trained YOLOv8 models (trained on COCO) struggle to detect small vehicles from a bird's-eye view. Fine-tuning on VisDrone allows the model to reliably detect cars, buses, trucks, and motorbikes even when they appear tiny or distorted by drone camera angles.

### Performance Metrics & Architecture Comparison
To ensure we were utilizing the best possible model for aerial traffic monitoring, we conducted a rigorous comparative experiment between **YOLOv8s** (Baseline) and the newer **YOLO11s** (Upgraded Architecture). Both models were trained for 50 epochs on the VisDrone dataset under identical hyperparameters.

| Metric | YOLOv8s (Baseline) | YOLO11s (New) | Difference (Δ) |
| :--- | :---: | :---: | :---: |
| **Precision** | **68.2%** | 61.9% | -6.31% |
| **Recall** | **53.3%** | 50.9% | -2.35% |
| **mAP@50** | **57.6%** | 55.0% | -2.68% |
| **mAP@50-95** | **34.9%** | 33.0% | -1.90% |

#### 🏆 Winner: YOLOv8s
Contrary to initial expectations, the YOLO11s architecture yielded a slight decrease in overall performance across all metrics compared to the YOLOv8s baseline. This outcome suggests that while YOLO11 incorporates advanced architectural refinements for general object detection, the mature feature-extraction backbone of **YOLOv8s** is naturally better suited to adapting to the unique top-down perspectives and small object scales of the VisDrone dataset.

Consequently, the **YOLOv8s** weights (`yolov8s-visdrone.pt`) have been retained and integrated into the primary video processing pipeline for final production deployment.

*(Note: While a Recall of ~53% may appear moderate in standard terrestrial applications, it is considered highly competitive for raw, unenhanced drone footage where vehicles appear extremely small. Our system mitigates this limitation in production by applying Real-ESRGAN Super-Resolution prior to inference).*

---
*Built for the future of Smart City infrastructure.*
