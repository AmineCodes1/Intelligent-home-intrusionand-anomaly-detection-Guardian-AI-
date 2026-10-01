"""Configuration settings for the intrusion detection system."""

IOT_CONFIG = {
    "sensors": {
        "vibration": {"min": 0, "max": 100, "normal_range": (0, 30), "unit": "m/s2"},
        "audio": {"min": 0, "max": 120, "normal_range": (20, 60), "unit": "dB"},
        "temperature": {"min": -10, "max": 50, "normal_range": (18, 28), "unit": "C"},
        "co2": {"min": 300, "max": 5000, "normal_range": (400, 1000), "unit": "ppm"},
        "pir_motion": {"min": 0, "max": 1, "normal_range": (0, 0), "unit": "binary"}
    },
    "sampling_rate": 1.0,
}

CAMERA_CONFIG = {
    "resolution": (640, 480),
    "fps": 15,
    "recording_duration": 10,
    "confidence_threshold": 0.5,
    "tracking_enabled": True
}

SEQUENCE_CONFIG = {
    "input_sequence_length": 10,
    "prediction_horizon": 5,
    "hidden_dim": 32,
    "num_layers": 1,
    "dropout": 0.0,
    "learning_rate": 0.001,
    "batch_size": 64,
    "num_workers": 0,
    "weight_decay": 0.0,
    "epochs": 10,
    "num_threads": 1,
    "threshold_percentile": 95.0,
    "startup_warmup_readings": 10,
    "reject_non_finite_readings": True,
    "reject_out_of_range_readings": True,
    "random_state": 42
}

SYSTEM_CONFIG = {
    "log_dir": "logs",
    "output_dir": "outputs",
    "model_dir": "models/saved",
    "data_dir": "data"
}
