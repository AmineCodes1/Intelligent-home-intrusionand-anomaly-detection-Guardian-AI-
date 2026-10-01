"""IoT data processing pipeline."""
import numpy as np
import pandas as pd
from typing import Dict
from collections import deque
from utils.config import IOT_CONFIG


class IoTProcessor:
    """Real-time IoT data processing pipeline."""
    
    def __init__(self, window_size=10):
        self.config = IOT_CONFIG["sensors"]
        self.window_size = window_size
        self.data_buffer = {sensor: deque(maxlen=window_size) for sensor in self.config.keys()}
        self.event_history = []
        
    def process_reading(self, reading: Dict[str, float]) -> Dict:
        """Buffer a reading and expose rolling features for downstream models."""
        for sensor in self.config.keys():
            if sensor in reading:
                self.data_buffer[sensor].append(reading[sensor])
        return {"reading": reading, "features": self._extract_features(reading)}
    
    def _extract_features(self, reading: Dict[str, float]) -> Dict[str, float]:
        """Extract features for ML classification."""
        features = {}
        
        for sensor in self.config.keys():
            if sensor in reading:
                value = reading[sensor]
                features[f"{sensor}_current"] = value
                
                if len(self.data_buffer[sensor]) >= 2:
                    buffer_list = list(self.data_buffer[sensor])
                    features[f"{sensor}_mean"] = np.mean(buffer_list)
                    features[f"{sensor}_std"] = np.std(buffer_list)
                    features[f"{sensor}_max"] = np.max(buffer_list)
                    features[f"{sensor}_min"] = np.min(buffer_list)
                    features[f"{sensor}_delta"] = buffer_list[-1] - buffer_list[-2]
                else:
                    features[f"{sensor}_mean"] = value
                    features[f"{sensor}_std"] = 0
                    features[f"{sensor}_max"] = value
                    features[f"{sensor}_min"] = value
                    features[f"{sensor}_delta"] = 0
        
        return features
    
    def get_feature_vector(self, reading: Dict[str, float]) -> np.ndarray:
        """Get feature vector for ML prediction."""
        features = self._extract_features(reading)
        return np.array(list(features.values())).reshape(1, -1)
    
    def get_raw_features(self, reading: Dict[str, float]) -> np.ndarray:
        """Get raw sensor values as feature vector."""
        values = [reading.get(sensor, 0) for sensor in self.config.keys()]
        return np.array(values).reshape(1, -1)
    
if __name__ == "__main__":
    from iot.simulator import IoTSimulator
    
    processor = IoTProcessor()
    simulator = IoTSimulator()
    
    print("Testing IoT feature buffering...")
    for i, (reading, state) in enumerate(simulator.generate_stream(n_samples=5, include_anomalies=True)):
        print(f"\n--- Reading {i+1} (True state: {state}) ---")
        result = processor.process_reading(reading)
        print(f"Features: {len(result['features'])}")
