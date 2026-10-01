"""Simulated IoT sensor data generator."""
import numpy as np
import time
from typing import Generator, Dict, Tuple
from utils.config import IOT_CONFIG


class IoTSimulator:
    """Simulates IoT sensor data for testing the intrusion detection system."""
    
    def __init__(self, seed=None):
        # Use None seed for real randomness during live testing
        self.rng = np.random.default_rng(seed)
        self.config = IOT_CONFIG["sensors"]
        self.current_state = "normal"
        self.intrusion_probability = 0.1
        
    def _generate_normal_reading(self, sensor: str) -> float:
        """Generate a normal reading for a sensor with added noise."""
        cfg = self.config[sensor]
        min_val, max_val = cfg["normal_range"]
        
        if sensor == "pir_motion":
            # 5% chance of false positive in normal state
            return float(self.rng.choice([0, 1], p=[0.95, 0.05]))
        else:
            mean = (min_val + max_val) / 2
            # Add some jitter to the normal range
            std = (max_val - min_val) / 4
            value = self.rng.normal(mean, std)
            # Add extra high-frequency noise
            noise = self.rng.uniform(-0.5, 0.5)
            return float(np.clip(value + noise, cfg["min"], cfg["max"]))
    
    def _generate_anomaly_reading(self, sensor: str) -> float:
        """Generate an anomalous reading (random spikes/drops)."""
        cfg = self.config[sensor]
        
        if sensor == "pir_motion":
            return float(self.rng.choice([0, 1], p=[0.4, 0.6]))
        else:
            anomaly_type = self.rng.choice(["spike", "drop", "drift", "noise"])
            normal_min, normal_max = cfg["normal_range"]
            
            if anomaly_type == "spike":
                return float(self.rng.uniform(normal_max * 1.1, cfg["max"]))
            elif anomaly_type == "drop":
                return float(self.rng.uniform(cfg["min"], normal_min * 0.9))
            elif anomaly_type == "drift":
                # Simulate a drifting sensor
                return float(self.rng.normal(normal_max * 1.05, 2.0))
            else:
                # Heavy noise
                return float(self.rng.uniform(cfg["min"], cfg["max"]))
    
    def generate_single_reading(self, include_anomaly=False) -> Tuple[Dict[str, float], str]:
        """Generate a sensor reading with optional non-human sensor faults."""
        if include_anomaly and self.rng.random() < 0.5:
            self.current_state = "anomaly"
        else:
            self.current_state = "normal"
        
        reading = {"timestamp": time.time()}
        
        for sensor in self.config.keys():
            if self.current_state == "normal":
                reading[sensor] = self._generate_normal_reading(sensor)
            else:
                reading[sensor] = self._generate_anomaly_reading(sensor)
        
        return reading, self.current_state
    
    def generate_stream(self, n_samples=100, include_anomalies=False) -> Generator:
        """Generate a stream of sensor readings and optional sensor anomalies."""
        for _ in range(n_samples):
            reading, state = self.generate_single_reading(include_anomaly=include_anomalies)
            yield reading, state
            time.sleep(0.01)

    def generate_normal_sensor_stream(self, n_samples=1000) -> np.ndarray:
        """Generate only ordered normal sensor values for self-supervised training."""
        readings = []
        for _ in range(n_samples):
            reading, _ = self.generate_single_reading(include_anomaly=False)
            readings.append([reading[sensor] for sensor in self.config.keys()])
        return np.asarray(readings, dtype=np.float32)

    def generate_evaluation_stream(self, block_length=30, n_blocks=20) -> Tuple[np.ndarray, np.ndarray]:
        """Generate alternating normal and anomalous blocks for balanced temporal evaluation."""
        readings = []
        labels = []
        for block_index in range(n_blocks):
            is_anomaly = block_index % 2 == 1
            for _ in range(block_length):
                if is_anomaly:
                    reading, _ = self.generate_single_reading(include_anomaly=False)
                    if block_index % 4 == 1:
                        subtle_sensor = list(self.config.keys())[block_index % len(self.config)]
                        offsets = {
                            "vibration": 8.0,
                            "audio": 12.0,
                            "temperature": 4.0,
                            "co2": 250.0,
                            "pir_motion": 1.0
                        }
                        cfg = self.config[subtle_sensor]
                        reading[subtle_sensor] = float(np.clip(
                            reading[subtle_sensor] + offsets[subtle_sensor],
                            cfg["min"], cfg["max"]
                        ))
                    else:
                        reading = {
                            sensor: self._generate_anomaly_reading(sensor)
                            for sensor in self.config.keys()
                        }
                    label = 1
                else:
                    reading, _ = self.generate_single_reading(include_anomaly=False)
                    label = 0
                readings.append([reading[sensor] for sensor in self.config.keys()])
                labels.append(label)
        return np.asarray(readings, dtype=np.float32), np.asarray(labels, dtype=np.int64)
