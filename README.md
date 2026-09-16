# Cricket DRS Service

A computer-vision-powered **Decision Review System (DRS)** backend for cricket video analysis.

The Khelaao DRS Service processes cricket footage to detect and track the ball, estimate its trajectory, identify key events such as bounce and impact, and generate broadcast-style replay videos.

Built with **Python, FastAPI, OpenCV, NumPy, and YOLO**.

---

## ✨ Features

* 🎯 **Cricket Ball Detection**

  * Optional YOLO-based ball detection
  * OpenCV and optical-flow based detection recovery

* 📍 **Ball Tracking**

  * Kalman-filter-based tracking
  * Gap filling when the ball temporarily disappears
  * Smooth trajectory estimation

* 📈 **Trajectory Analysis**

  * Ball trajectory visualization
  * Future-path projection
  * Bounce and impact detection
  * Wicket/stump visualization

* 🎥 **Video Processing**

  * Automatic video stabilization
  * Slow-motion replay generation
  * Zoom stages for replay sequences
  * Minimum 720p output rendering

* 🔊 **Replay Audio**

  * Optional broadcast-style replay audio/stinger

* ⚡ **REST API**

  * FastAPI backend
  * Video processing endpoints
  * Static replay-video serving

---

## 🧠 Processing Pipeline

```text
Cricket Video
      │
      ▼
Video Upload
      │
      ▼
Ball Detection
(YOLO / OpenCV)
      │
      ▼
Optical Flow Recovery
      │
      ▼
Kalman Tracking
      │
      ▼
Trajectory Smoothing
      │
      ▼
Event Detection
(Bounce / Impact / Wicket)
      │
      ▼
Visualization & Projection
      │
      ▼
Replay Generation
      │
      ▼
Processed Video
```

---

## 🛠️ Tech Stack

| Technology         | Purpose                                  |
| ------------------ | ---------------------------------------- |
| Python             | Core backend                             |
| FastAPI            | REST API                                 |
| Uvicorn            | ASGI server                              |
| OpenCV             | Computer vision & video processing       |
| NumPy              | Numerical processing                     |
| YOLO / Ultralytics | Optional cricket-ball detection          |
| FFmpeg             | Video encoding and browser compatibility |
| Kalman Filter      | Ball tracking                            |

The current project dependencies include FastAPI, Uvicorn, OpenCV, NumPy, Python Multipart, and Ultralytics.

---

## 📁 Project Structure

```text
Drs-Service/
│
├── app/
│   ├── api/
│   ├── core/
│   └── ...
│
├── drs/
│   └── ...
│
├── models/
│   └── cricket-ball.pt
│
├── public/
│   └── videos/
│
├── tmp/
│
├── main.py
├── requirements.txt
├── setup.ps1
├── .env.drs.example
├── DRS_SETUP.md
└── README.md
```

---

# 🚀 Getting Started

## Prerequisites

Make sure you have:

* Python 3.10+
* FFmpeg
* Git
* Windows PowerShell (for the automated setup script)

FFmpeg is recommended because generated replay videos are re-encoded into browser-compatible H.264.

---

## 1. Clone the Repository

```bash
git clone https://github.com/SAADSAEED007/Drs-Service.git
cd Drs-Service
```

---

## 2. Run Automated Setup

On Windows PowerShell:

```powershell
.\setup.ps1
```

The setup script automatically:

* Creates the virtual environment
* Installs Python dependencies
* Creates required directories
* Checks FFmpeg availability
* Creates the DRS environment configuration

---

## 3. Activate Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

---

## 4. Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

The service will be available at:

```text
http://localhost:8001
```

FastAPI automatically provides interactive API documentation at:

```text
http://localhost:8001/docs
```

---

# 🎯 Optional YOLO Model

The service can use a trained YOLO model for cricket-ball detection.

Place your model inside:

```text
models/cricket-ball.pt
```

Or configure a custom model path:

### Windows

```powershell
$env:DRS_YOLO_MODEL="C:\path\to\cricket-ball.pt"
```

The service can also look for:

```text
models/cricket-ball.pt
models/yolov8n.pt
```

If no YOLO model is available, the pipeline can continue using OpenCV and optical-flow-based detection.

---

# ⚙️ Configuration

The service supports environment-based configuration.

### Disable Video Stabilization

```powershell
$env:DRS_ENABLE_STABILIZATION="0"
```

### Disable Replay Audio

```powershell
$env:DRS_ENABLE_REPLAY_AUDIO="0"
```

### Custom Models Directory

```powershell
$env:DRS_MODELS_DIRECTORY="C:\path\to\models"
```

---

# 🎥 Video Processing

The DRS pipeline performs multiple stages of analysis:

1. Upload cricket footage
2. Detect the cricket ball
3. Recover missing detections using optical flow
4. Track the ball using Kalman filtering
5. Smooth the detected trajectory
6. Estimate future ball movement
7. Detect bounce and impact events
8. Generate visual overlays
9. Stabilize the video
10. Create a broadcast-style replay
11. Export the processed video

The current setup documentation describes these stages, including YOLO detection, optical-flow recovery, Kalman tracking, trajectory projection, bounce/impact visualization, and replay generation.

---

# 📡 API

The application is built using FastAPI and exposes its application routes through the project's API router. Generated videos are served through the `/videos` path.

Once the server is running, open:

```text
http://localhost:8001/docs
```

to explore and test the available endpoints through Swagger UI.

---

# 📊 Accurac
