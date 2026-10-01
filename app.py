import streamlit as st
import pandas as pd
import numpy as np
import time
import cv2
import os
import joblib
from datetime import datetime
from PIL import Image
import sys

# Ensure Project_Amine is in path for imports
if os.path.exists("Project_Amine"):
    sys.path.append(os.path.abspath("Project_Amine"))
    sys.path.append(os.getcwd())
elif os.path.exists("Project-Amine"):
    sys.path.append(os.path.abspath("Project-Amine"))
    sys.path.append(os.getcwd())

# Import logic using relative paths
try:
    from iot.simulator import IoTSimulator
    from camera.video_simulator import VideoSimulator
    from camera.detector import CameraDetector
    from utils.config import IOT_CONFIG, CAMERA_CONFIG
    from anomalies.lstm_detector import LSTMSequencePredictor
except ImportError:
    try:
        from Project_Amine.iot.simulator import IoTSimulator
        from Project_Amine.camera.video_simulator import VideoSimulator
        from Project_Amine.camera.detector import CameraDetector
        from Project_Amine.utils.config import IOT_CONFIG, CAMERA_CONFIG
        from Project_Amine.anomalies.lstm_detector import LSTMSequencePredictor
    except ImportError as e:
        st.error(f"Erreur d'importation : {e}")
        st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="GuardianAI | Enterprise Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced UI Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: #00f2fe;
        --secondary: #4facfe;
        --bg-dark: #06090f;
        --card-bg: rgba(13, 17, 23, 0.8);
        --border: rgba(255, 255, 255, 0.08);
        --accent-red: #ff4d4d;
        --accent-orange: #ffa500;
        --accent-green: #00ff88;
        --glow: 0 0 20px rgba(79, 172, 254, 0.25);
    }

    /* Global Overrides */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1c2533, #06090f) !important;
        color: #f0f6fc !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Professional Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid var(--border) !important;
    }

    /* Fix visibility of text and labels in sidebar */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCheckbox p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }

    /* Contrast for Expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: transparent !important;
        border: 1px solid var(--border) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background-color: transparent !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }

    /* Slider values and text */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
        color: var(--secondary) !important;
        font-weight: 700 !important;
    }

    /* Force all text inside sidebar to be white/readable */
    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    /* Exceptions for specific accents */
    [data-testid="stSidebar"] .stSlider * {
        color: inherit;
    }

    /* Metric Styling for high visibility */
    [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        color: #00f2fe !important; /* Bright Cyan */
        text-shadow: 0 0 15px rgba(0, 242, 254, 0.5);
    }
    
    [data-testid="stMetricLabel"] p {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        margin-bottom: 5px !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.3s ease !important;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-5px) !important;
        border-color: var(--secondary) !important;
        background: rgba(79, 172, 254, 0.08) !important;
    }

    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--secondary);
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Model Badge */
    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        background: rgba(79, 172, 254, 0.15);
        color: var(--secondary);
        border: 1px solid rgba(79, 172, 254, 0.3);
    }
    
    .best-badge {
        background: rgba(0, 255, 136, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(0, 255, 136, 0.3);
    }

    /* Metric Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 10px;
    }

    .mini-metric {
        padding: 10px;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        text-align: center;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_ai_models():
    sequence_predictor = LSTMSequencePredictor()
    if not sequence_predictor.load():
        st.error("LSTM model not found. Run `python main.py --train` first.")
        st.stop()
    # Initialize Camera Detector for person detection & tracking
    cam_detector = CameraDetector()
    return sequence_predictor, cam_detector

sequence_predictor, cam_detector = load_ai_models()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color:#4facfe;'>COMMAND CENTER</h2>", unsafe_allow_html=True)
    
    with st.expander("📡 IOT TELEMETRY", expanded=True):
        st.markdown("**Automatic IoT simulation**")
        st.caption("100 generated sensor readings per scan")

    with st.expander("🧠 AI PROTOCOLS", expanded=True):
        include_intruder = st.toggle("Human Signature Simulation", value=False)
        duration = 100

    launch_btn = st.button("EXECUTE NEURAL SCAN", width='stretch', type="primary")

# --- Header ---
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:30px; background:rgba(255,255,255,0.02); padding:25px; border-radius:15px; border:1px solid rgba(255,255,255,0.05);">
    <div>
        <h1 style="margin:0; font-size:2.8rem;">GuardianAI <span style="color:rgba(255,255,255,0.2)">v2.0</span></h1>
        <p style="margin:0; color:rgba(255,255,255,0.4);">Advanced Neural Surveillance & Intrusion Intelligence</p>
    </div>
    <div style="text-align:right;">
        <span style="background:#00ff8822; color:#00ff88; padding:5px 15px; border-radius:10px; font-weight:700; font-size:0.8rem; border:1px solid #00ff8844;">SYSTEM READY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Layout ---
col_main, col_side = st.columns([2, 1], gap="large")

with col_main:
    st.markdown('<div class="section-title">Visual Intelligence Stream</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card" style="padding:10px;">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sensor Prediction Monitor</div>', unsafe_allow_html=True)
    sensor_placeholder = st.empty()

with col_side:
    st.markdown('<div class="section-title">Threat Assessment</div>', unsafe_allow_html=True)
    status_placeholder = st.empty()
    
    st.markdown('<div class="section-title">Model Attribution</div>', unsafe_allow_html=True)
    attribution_placeholder = st.empty()

    st.markdown('<div class="section-title">Event Timeline</div>', unsafe_allow_html=True)
    terminal_placeholder = st.empty()

# --- Logic ---
if 'terminal_logs' not in st.session_state:
    st.session_state.terminal_logs = []
if 'prediction_errors' not in st.session_state:
    st.session_state.prediction_errors = []

def add_smart_log(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {"info": "#4facfe", "warning": "#ffa500", "error": "#ff4d4d", "success": "#00ff88"}
    line = f'<div style="color:{colors[level]}; font-family:monospace; margin-bottom:5px;">[{ts}] {msg}</div>'
    st.session_state.terminal_logs.append(line)
    st.session_state.terminal_logs = st.session_state.terminal_logs[-15:]

if launch_btn:
    st.session_state.terminal_logs = []
    add_smart_log("Simulation initiated...", "info")
    
    # Reset camera detector state for new run
    if cam_detector.tracker:
        cam_detector.tracker.reset()
    cam_detector.recording = False
    cam_detector.recording_buffer = []
    sequence_predictor.reset_stream()
    st.session_state.prediction_errors = []

    prog = st.progress(0)
    video_sim = VideoSimulator()
    sensor_sim = IoTSimulator()
    sensor_stream = sensor_sim.generate_stream(n_samples=duration, include_anomalies=True)
    sensor_names = list(IOT_CONFIG["sensors"].keys())

    for i, (sensor_reading, sensor_state) in enumerate(sensor_stream):
        # Get frame with optional intruder
        raw_frame, sim_detections = video_sim.get_frame(include_intruder=include_intruder)
        
        # Process frame with YOLO + Tracking
        # CameraDetector handles YOLO detections and real DeepSORT tracking.
        cam_result = cam_detector.process_frame(raw_frame, sim_detections)
        
        # Annotate frame for display
        display_frame = cam_detector.draw_detections(raw_frame, cam_result)
        
        input_data = [sensor_reading[sensor] for sensor in sensor_names]
        sensor_placeholder.dataframe(
            pd.DataFrame([input_data], columns=sensor_names),
            hide_index=True,
            use_container_width=True
        )
        
        sequence_result = sequence_predictor.observe(np.array(input_data, dtype=np.float32))
        sequence_status = sequence_result.get("status", "forecast_pending")
        is_anomaly_detected = sequence_result["ready"] and sequence_result["is_anomaly"]
        if sequence_result["ready"]:
            st.session_state.prediction_errors.append(sequence_result["prediction_error"])
            st.session_state.prediction_errors = st.session_state.prediction_errors[-100:]
        
        # Smart Decision Fusion
        is_human = cam_result["is_intrusion"]
        # Comprehensive status logic
        status_parts = []
        if is_human:
            status_parts.append("INTRUSION HUMAINE 👤")
        
        is_intrusion = len(status_parts) > 0
        
        if is_intrusion and is_anomaly_detected:
            f_s = " + ".join(status_parts) + " (FUTURE ANOMALY)"
        elif is_intrusion:
            f_s = " + ".join(status_parts)
        elif is_anomaly_detected:
            f_s = "FUTURE ANOMALY ⚠️"
        elif sequence_status in {"warming_up", "invalid_reading"}:
            f_s = "SENSOR WARMUP"
        elif not sequence_result["ready"]:
            f_s = "WAITING FOR FUTURE"
        else:
            f_s = "MAISON SÉCURISÉE 🟢"
        
        clr = "#00ff88" if "SÉCURISÉE" in f_s else "#ff4d4d" if is_intrusion else "#ffa500"

        # Update Visuals
        video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), width='stretch')
        
        # Update Status
        diag_html = ""
        if is_human:
            p_count = cam_result["person_count"]
            diag_html += f'<div style="font-size:0.8rem; color:#ff4d4d; margin-top:10px; font-weight:600;">👤 DETECTION: {p_count} humain(s) (YOLOv8 + DeepSort)</div>'
        if sequence_result["ready"]:
            diag_html += f'<div style="font-size:0.8rem; color:#ffa500; margin-top:5px; font-weight:600;">Sensor state: {sensor_state.upper()} | Prediction error: {sequence_result["prediction_error"]:.6f} | Threshold: {sequence_result["anomaly_threshold"]:.6f}</div>'
        elif sequence_status == "invalid_reading":
            diag_html += f'<div style="font-size:0.8rem; color:#ffa500; margin-top:5px; font-weight:600;">Sensor reading ignored: {sequence_result.get("reason", "invalid input")}. No anomaly alert emitted.</div>'
        elif sequence_status == "warming_up":
            diag_html += f'<div style="font-size:0.8rem; color:#4facfe; margin-top:5px; font-weight:600;">Warmup: {sequence_result["valid_readings"]}/{sequence_result["startup_warmup_readings"]} valid readings. Alerts disabled.</div>'
        else:
            diag_html += f'<div style="font-size:0.8rem; color:#4facfe; margin-top:5px; font-weight:600;">Sensor state: {sensor_state.upper()} | Prediction issued; awaiting {sequence_predictor.prediction_horizon} actual future readings.</div>'
        
        # Notifications
        if (is_intrusion or is_anomaly_detected) and i % 20 == 0:
            st.toast(f"🚨 {f_s}", icon="🛡️")
            add_smart_log(f"ALERT DISPATCHED: {f_s}", "error")

        status_placeholder.markdown(f"""
        <div class="glass-card" style="text-align:center; border-color:{clr}44;">
            <div style="font-size:0.7rem; color:rgba(255,255,255,0.4); margin-bottom:10px;">THREAT LEVEL</div>
            <div style="font-size:1.8rem; font-weight:900; color:{clr};">{f_s}</div>
            {diag_html}
            <div style="font-size:0.8rem; margin-top:10px;">Intrusion source: video detector</div>
        </div>
        """, unsafe_allow_html=True)

        # Update Attribution
        attribution_placeholder.markdown(f"""
        <div class="glass-card" style="background:rgba(0,255,136,0.05); border-color:#00ff8833;">
            <div style="font-size:0.7rem; color:rgba(255,255,255,0.4); margin-bottom:5px;">INTRUSION DETECTOR</div>
            <div style="font-size:1.1rem; font-weight:700; color:#00ff88;">YOLOv8 + tracker</div>
            <div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:5px;">Video presence is the intrusion signal.</div>
        </div>
        """, unsafe_allow_html=True)

        # Update Terminal
        terminal_placeholder.markdown(f"""
        <div style="height:200px; overflow-y:auto; background:rgba(0,0,0,0.3); padding:10px; border-radius:5px; border:1px solid rgba(255,255,255,0.05);">
            {''.join(st.session_state.terminal_logs[::-1])}
        </div>
        """, unsafe_allow_html=True)

        prog.progress((i+1)/duration)
        time.sleep(0.01)

    # Save final recording if any intrusion happened
    if cam_detector.recording:
        # The CameraDetector._stop_recording() handles drawing detections on all frames in the video
        cam_detector._stop_recording()
        # Find the latest recording
        recordings_dir = "outputs/recordings"
        if os.path.exists(recordings_dir):
            files = [os.path.join(recordings_dir, f) for f in os.listdir(recordings_dir) if f.endswith(".avi")]
            if files:
                latest_recording = max(files, key=os.path.getctime)
                add_smart_log(f"Protocol: Intrusion archive generated at {latest_recording}", "success")
                st.info(f"Dernier enregistrement (avec tracking) : {latest_recording}")

    # --- Neural Performance Metrics ---
    if sequence_result["ready"]:
        st.markdown('<div class="section-title">Future Sequence Comparison</div>', unsafe_allow_html=True)
        comparison = pd.DataFrame(
            np.column_stack((sequence_result["actual_sequence"], sequence_result["predicted_sequence"])),
            columns=[f"{sensor}_actual" for sensor in IOT_CONFIG["sensors"]] + [f"{sensor}_predicted" for sensor in IOT_CONFIG["sensors"]]
        )
        st.line_chart(comparison)
        error_frame = pd.DataFrame({"prediction_error": st.session_state.prediction_errors})
        error_frame["anomaly_threshold"] = sequence_result["anomaly_threshold"]
        st.line_chart(error_frame)
