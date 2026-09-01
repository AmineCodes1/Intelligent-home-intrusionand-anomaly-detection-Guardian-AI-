# GuardianAI – Smart Anti-Intrusion System

GuardianAI is a hybrid IoT + computer-vision stack for home/enterprise intrusion detection. It fuses simulated sensor telemetry, multi-model ML classifiers, anomaly detection (Isolation Forest, Autoencoder, LSTM), and YOLOv8-based human detection with DeepSort-style tracking. A Streamlit command center provides a live dashboard and simulation controls.

## Features
- Simulated IoT sensors (vibration, audio, temperature, CO₂, PIR) with intrusion/anomaly generation.
- Multi-model intrusion classifier (Random Forest, SVM, XGBoost) with saved artifacts for reuse.
- Anomaly ensemble (Isolation Forest, Autoencoder, LSTM) on point and sequence data.
- Camera pipeline with YOLOv8 (or simulated fallback) plus DeepSort-style tracker; auto-records intrusion clips.
- Streamlit "GuardianAI" control panel for running combined video + IoT simulations.
- CLI demos for training, IoT monitoring, camera detection, or full integrated run.

## Project Structure
```
Project_Amine/
├─ app.py                # Streamlit UI entrypoint
├─ main.py               # CLI orchestrator (train/demo)
├─ yolov8n.pt            # YOLOv8n weights (used if ultralytics installed)
├─ anomalies/            # IsolationForest, Autoencoder, LSTM ensemble
├─ camera/               # YOLO detector, DeepSort-like tracker, video simulator
├─ iot/                  # Sensor simulator and processor
├─ models/               # Intrusion classifier + saved models
├─ outputs/recordings/   # Saved intrusion clips
├─ logs/                 # Runtime logs
└─ utils/                # Config and logging utilities
```

## Quickstart
1) **Environment**
```
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install streamlit pandas numpy opencv-python-headless pillow colorama ultralytics filterpy scipy scikit-learn xgboost joblib
```
(If you need GUI OpenCV, replace `opencv-python-headless` with `opencv-python`.)

2) **Run Streamlit Command Center**
```
streamlit run app.py
```

3) **CLI Demos**
```
python main.py --train      # train all ML/anomaly models
python main.py --iot        # IoT monitoring demo
python main.py --camera     # camera intrusion demo
python main.py --full       # full pipeline (train + IoT + camera)
```

## How It Works
- **Config**: Tuning lives in `utils/config.py` (sensor ranges, camera FPS, ML hyperparams, thresholds).
- **Training**: `main.py --train` simulates labeled data, trains RF/SVM/XGBoost + anomaly models, and saves them to `models/saved` for later inference.
- **Inference**: Classifier and anomaly ensemble are loaded at runtime; camera detector uses YOLOv8 if available, otherwise a simulated detector.
- **Recording**: When people are detected, frames are annotated and clips are written under `outputs/recordings`.
- **Logging**: Structured console + file logging under `logs`.

## Data & Models
- Pretrained artifacts are stored in `models/saved` (joblib files). Regenerate anytime via `--train`.
- Video outputs and sample frames are stored in `outputs/`.
- No external datasets are required; data is simulated.

## Troubleshooting
- If YOLO weights fail to load, the detector automatically uses a simulation; install `ultralytics` (and PyTorch) to enable real YOLOv8 inference.
- For OpenCV video writing on Linux/macOS, you may need additional codecs; on Windows the bundled XVID should work.
- If Streamlit cannot import local modules, ensure you run from the project root so relative imports resolve.

## License
This project is licensed under the MIT License.
