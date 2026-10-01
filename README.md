# GuardianAI – Smart Anti-Intrusion System

GuardianAI is a hybrid IoT + computer-vision stack for home/enterprise intrusion detection. Video detects human intrusion, while one LSTM next-sequence detector monitors unusual sensor behavior. A Streamlit command center provides a live dashboard and simulation controls.

## Features
- Simulated IoT sensors (vibration, audio, temperature, CO₂, PIR) with intrusion/anomaly generation.
- LSTM next-sequence anomaly detection trained only on normal sensor streams.
- Camera pipeline with YOLOv8 (or simulated fallback) plus real DeepSORT tracking; auto-records intrusion clips.
- Streamlit "GuardianAI" control panel for running combined video + IoT simulations.
- CLI demos for training, IoT monitoring, camera detection, or full integrated run.

## Project Structure
```
Project_Amine/
├─ app.py                # Streamlit UI entrypoint
├─ main.py               # CLI orchestrator (train/demo)
├─ yolov8n.pt            # YOLOv8n weights (used if ultralytics installed)
├─ anomalies/            # LSTM history-to-future prediction and thresholding
├─ camera/               # YOLO detector, DeepSORT tracker, video simulator
├─ iot/                  # Sensor simulator and processor
├─ models/               # Saved LSTM preprocessing/model artifacts
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
pip install -r requirements.txt
```
(If you need GUI OpenCV, replace `opencv-python-headless` with `opencv-python`.)

2) **Run Streamlit Command Center**
```
streamlit run app.py
```

3) **CLI Demos**
```
python main.py --train      # train the LSTM anomaly detector
python main.py --iot        # IoT monitoring demo
python main.py --camera     # camera intrusion demo
python main.py --full       # full pipeline (train + IoT + camera)
```

## How It Works
- **Config**: Tuning lives in `utils/config.py` (sensor ranges, camera FPS, LSTM hyperparameters, thresholds).
- **Training**: `main.py --train` trains an encoder/decoder LSTM on normal sensor histories, predicts five future steps, and derives its error threshold from normal validation windows only. Intrusion classification is handled by the video detector, not sensor classifiers.
- **Inference**: The dashboard uses YOLOv8/tracking for human intrusion and separately consumes a 100-reading `IoTSimulator` stream, predicts the next five sensor readings, waits for those five actual readings, and compares scaled mean-squared prediction error with the saved threshold.
- **Recording**: When people are detected, frames are annotated and clips are written under `outputs/recordings`.
- **Logging**: Structured console + file logging under `logs`.

## Data & Models
- Trained artifacts are stored in `models/saved` (`lstm_next_sequence.pt` and its scaler). Regenerate anytime via `--train`.
- Video outputs and sample frames are stored in `outputs/`.
- No external datasets are required; data is simulated.

## Troubleshooting
- If YOLO weights fail to load, the detector automatically uses a simulation; install `ultralytics` (and PyTorch) to enable real YOLOv8 inference.
- For OpenCV video writing on Linux/macOS, you may need additional codecs; on Windows the bundled XVID should work.
- If Streamlit cannot import local modules, ensure you run from the project root so relative imports resolve.

## LSTM anomaly-detection architecture

```text
five sensor values -> StandardScaler -> ten-step history -> encoder LSTM
	-> decoder LSTM -> predicted next five sensor sequences
	-> actual next five sequences -> scaled MSE prediction error
	-> normal-validation percentile threshold -> NORMAL / ANOMALY
```

The model never reconstructs its input and never sees intrusion labels during training. Sliding windows preserve temporal order: `[t..t+9]` is paired with `[t+10..t+14]`, then the start advances by one step. Labels are used only for the final test metrics. Because the detector predicts the future, the dashboard reports a waiting state until the corresponding future observations exist.

## License
This project is licensed under the MIT License.
