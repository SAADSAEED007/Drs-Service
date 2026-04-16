# Khelaao DRS Upgrade Setup

## Quick start

Windows PowerShell:

```powershell
cd Khelaao-DRS-Service
.\setup.ps1
```

If you already have a trained cricket-ball model:

```powershell
.\setup.ps1 -YoloModelPath "C:\path\to\cricket-ball.pt"
```

Disable stabilization or replay audio during setup:

```powershell
.\setup.ps1 -DisableStabilization -DisableReplayAudio
```

The setup script will:

- create `.venv`
- install Python dependencies
- create `tmp`, `public/videos`, and `models`
- check whether `ffmpeg` is available
- generate `.env.drs` with DRS pipeline options

## 1. Install Python dependencies manually

```bash
cd Khelaao-DRS-Service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Install FFmpeg

FFmpeg is strongly recommended because replay videos are re-encoded to browser-safe H.264.

Windows:

```bash
winget install Gyan.FFmpeg
```

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install ffmpeg -y
```

macOS:

```bash
brew install ffmpeg
```

## 3. Optional YOLO ball detector

The detector supports YOLO automatically when a model file is present.

Set an environment variable pointing to your cricket-ball model:

```bash
set DRS_YOLO_MODEL=C:\path\to\cricket-ball.pt
```

Or edit `.env.drs` after running `setup.ps1`.

If `DRS_YOLO_MODEL` is not set, the service looks for:

- `models/cricket-ball.pt`
- `models/yolov8n.pt`

If no model is found, the pipeline still runs with the upgraded OpenCV + optical-flow detector.

You can also change the model directory:

```bash
set DRS_MODELS_DIRECTORY=C:\path\to\models
```

## 4. Optional pipeline toggles

Disable stabilization:

```bash
set DRS_ENABLE_STABILIZATION=0
```

Disable replay audio stinger:

```bash
set DRS_ENABLE_REPLAY_AUDIO=0
```

## 5. Run the FastAPI service

If you used `setup.ps1`, activate the virtual environment first:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

## 6. Processing flow

The upgraded pipeline now includes:

- optional YOLO ball detection
- optical-flow-assisted detection recovery
- automatic optical-flow video stabilization
- Kalman-based tracking with gap filling
- smooth fitted trajectory and future-path projection
- bounce, impact, and wicket visualization
- broadcast-style replay sequencing with slow-motion and zoom stages
- optional replay audio stinger during export
- 720p minimum output rendering

## Notes

- Best accuracy comes from bright daylight clips with the ball visible for multiple frames.
- For fast bowling, keep the camera steady and frame both bounce area and stumps.
- Without FFmpeg, replay files may still be produced but some browsers/players may not decode them correctly.
