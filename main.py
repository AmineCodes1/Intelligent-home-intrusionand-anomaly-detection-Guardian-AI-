"""
Smart Anti-Intrusion System for Connected Homes
Main entry point and system orchestration.
"""
import os
import sys
import time
import argparse
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import IOT_CONFIG, SYSTEM_CONFIG
from iot.simulator import IoTSimulator
from camera.detector import CameraDetector
from anomalies.lstm_detector import LSTMSequencePredictor


def ensure_directories():
    """Create necessary directories."""
    dirs = [
        SYSTEM_CONFIG["log_dir"],
        SYSTEM_CONFIG["output_dir"],
        SYSTEM_CONFIG["model_dir"],
        SYSTEM_CONFIG["data_dir"],
        "outputs/recordings"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_banner():
    """Print system banner."""
    banner = f"""
{Fore.CYAN}{'='*70}
     SMART ANTI-INTRUSION SYSTEM FOR CONNECTED HOMES
              IoT + Video Recognition + ML
{'='*70}{Style.RESET_ALL}

{Fore.YELLOW}Features:{Style.RESET_ALL}
  - Real-time IoT sensor monitoring (vibration, audio, temp, CO2, PIR)
    - Camera-based human intrusion detection with YOLOv8 and real DeepSORT tracking
    - LSTM next-sequence sensor anomaly detection trained on normal behavior
  - Automatic video recording on intrusion detection

{Fore.GREEN}Status: System Initializing...{Style.RESET_ALL}
"""
    print(banner)


def train_models():
    """Train the LSTM sensor anomaly model."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("TRAINING MACHINE LEARNING MODELS")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    simulator = IoTSimulator()
    
    print("1. Training LSTM Next-Sequence Anomaly Detector...")
    normal_train = simulator.generate_normal_sensor_stream(n_samples=1600)
    normal_validation = simulator.generate_normal_sensor_stream(n_samples=600)
    test_stream, test_labels = simulator.generate_evaluation_stream(block_length=30, n_blocks=40)
    predictor = LSTMSequencePredictor()
    sequence_results = predictor.train(normal_train, normal_validation)
    predictor.save()
    evaluation = predictor.evaluate(test_stream, test_labels)
    print(f"   - Input shape: (batch, {sequence_results['input_sequence_length']}, {normal_train.shape[1]})")
    print(f"   - Target shape: (batch, {sequence_results['prediction_horizon']}, {normal_train.shape[1]})")
    print(f"   - Normal validation threshold: {sequence_results['threshold']:.6f}")
    print("\n   Evaluation on labeled test stream:")
    print(f"   - Test points: {len(test_labels)} ({int(np.sum(test_labels == 0))} normal, {int(np.sum(test_labels == 1))} anomalous)")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        print(f"   - {metric.title()}: {evaluation[metric]:.4f}")
    print(f"   - Confusion matrix: {evaluation['confusion_matrix']}")
    
    print(f"\n{Fore.GREEN}All models trained and saved successfully!{Style.RESET_ALL}")
    
    return predictor


def run_iot_demo(predictor, duration=30):
    """Run IoT sensor monitoring demo."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("IOT SENSOR MONITORING DEMO")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    simulator = IoTSimulator()
    anomaly_count = 0
    predictor_feature_names = list(IOT_CONFIG["sensors"].keys())
    
    print(f"Monitoring for {duration} sensor readings...\n")
    
    for i, (reading, true_state) in enumerate(simulator.generate_stream(n_samples=duration, include_anomalies=True)):
        features = np.array([reading[sensor] for sensor in predictor_feature_names])
        sequence_result = predictor.observe(features)
        
        anomaly_status = "WAITING FOR FUTURE"
        if sequence_result["ready"]:
            anomaly_status = "ANOMALY" if sequence_result["is_anomaly"] else "NORMAL"
            print(f"   Prediction error: {sequence_result['prediction_error']:.6f} | Threshold: {sequence_result['anomaly_threshold']:.6f}")
        print(f"[{i+1:3d}] Sensor state: {true_state} | Sequence: {anomaly_status}")
        
        if sequence_result["ready"] and sequence_result["is_anomaly"]:
            anomaly_count += 1
        
        time.sleep(0.1)
    
    print(f"\n{Fore.YELLOW}Demo Summary:{Style.RESET_ALL}")
    print(f"  Total readings: {duration}")
    print(f"  Anomalies detected: {anomaly_count}")


def run_camera_demo():
    """Run camera-based detection demo."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("CAMERA-BASED INTRUSION DETECTION DEMO")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    detector = CameraDetector()
    
    print("Running simulation with 100 frames...")
    print("Intrusion occurs between frames 30-60\n")
    
    results = detector.run_simulation(n_frames=100, intrusion_start=30, intrusion_end=60)
    
    detector.save_sample_frames(results)
    
    intrusion_frames = sum(1 for r in results if r["is_intrusion"])
    
    print(f"\n{Fore.YELLOW}Camera Demo Summary:{Style.RESET_ALL}")
    print(f"  Total frames: {len(results)}")
    print(f"  Intrusion frames: {intrusion_frames}")
    print(f"  Recordings saved: {detector.intrusion_count}")
    print(f"  Sample frames saved to: outputs/")


def run_full_system():
    """Run the complete integrated system."""
    print_banner()
    ensure_directories()
    
    print(f"\n{Fore.YELLOW}Phase 1: Training Models{Style.RESET_ALL}")
    print("-" * 40)
    predictor = train_models()
    
    print(f"\n{Fore.YELLOW}Phase 2: IoT Monitoring Demo{Style.RESET_ALL}")
    print("-" * 40)
    run_iot_demo(predictor, duration=20)
    
    print(f"\n{Fore.YELLOW}Phase 3: Camera Detection Demo{Style.RESET_ALL}")
    print("-" * 40)
    run_camera_demo()
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print("SYSTEM DEMONSTRATION COMPLETE")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    print(f"""
{Fore.CYAN}Outputs generated:{Style.RESET_ALL}
  - Trained models saved in: models/saved/
  - Detection frames saved in: outputs/
  - Video recordings saved in: outputs/recordings/
  - System logs saved in: logs/

{Fore.CYAN}Available commands:{Style.RESET_ALL}
  python main.py --train     : Train all models
  python main.py --iot       : Run IoT monitoring demo
  python main.py --camera    : Run camera detection demo
  python main.py --full      : Run complete system demo (default)

{Fore.GREEN}System Status: HOME SECURED{Style.RESET_ALL}
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Smart Anti-Intrusion System")
    parser.add_argument("--train", action="store_true", help="Train all ML models")
    parser.add_argument("--iot", action="store_true", help="Run IoT monitoring demo")
    parser.add_argument("--camera", action="store_true", help="Run camera detection demo")
    parser.add_argument("--full", action="store_true", help="Run complete system demo")
    
    args = parser.parse_args()
    
    ensure_directories()
    
    if args.train:
        print_banner()
        train_models()
    elif args.iot:
        print_banner()
        predictor = LSTMSequencePredictor()
        if not predictor.load():
            print("LSTM model not found. Training first...")
            predictor = train_models()
        run_iot_demo(predictor)
    elif args.camera:
        print_banner()
        run_camera_demo()
    else:
        run_full_system()


if __name__ == "__main__":
    main()
