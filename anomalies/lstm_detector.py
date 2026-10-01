"""LSTM next-sequence prediction for unsupervised sensor anomaly detection."""
from collections import deque
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.config import SEQUENCE_CONFIG, SYSTEM_CONFIG
from utils.logger import logger


class SequencePredictorNetwork(nn.Module):
    """Encoder/decoder LSTM that predicts a future sequence, not its input."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 prediction_horizon: int, dropout: float):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.prediction_horizon = prediction_horizon
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=recurrent_dropout)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=recurrent_dropout)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, state = self.encoder(history)
        decoder_input = torch.zeros(history.shape[0], self.prediction_horizon, history.shape[2], device=history.device)
        decoded, _ = self.decoder(decoder_input, state)
        return self.output(decoded)


class LSTMSequencePredictor:
    """Train on normal history and classify future windows by prediction error."""

    def __init__(self):
        self.config = SEQUENCE_CONFIG
        self.input_sequence_length = self.config["input_sequence_length"]
        self.prediction_horizon = self.config["prediction_horizon"]
        self.hidden_dim = self.config["hidden_dim"]
        self.num_layers = self.config["num_layers"]
        self.model: Optional[SequencePredictorNetwork] = None
        self.scaler = StandardScaler()
        self.threshold: Optional[float] = None
        self.feature_count: Optional[int] = None
        self.is_trained = False
        self.model_path = os.path.join(SYSTEM_CONFIG["model_dir"], "lstm_next_sequence.pt")
        self.scaler_path = os.path.join(SYSTEM_CONFIG["model_dir"], "lstm_next_sequence_scaler.joblib")
        self.history = deque(maxlen=self.input_sequence_length)
        self.pending_prediction: Optional[np.ndarray] = None
        self.pending_actual = []
        os.makedirs(SYSTEM_CONFIG["model_dir"], exist_ok=True)

    def create_windows(self, stream: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create [history -> future] windows while preserving stream order."""
        stream = np.asarray(stream, dtype=np.float32)
        window_count = len(stream) - self.input_sequence_length - self.prediction_horizon + 1
        if window_count <= 0:
            raise ValueError("The stream is shorter than one input plus prediction window.")
        inputs = np.stack([stream[index:index + self.input_sequence_length] for index in range(window_count)])
        targets = np.stack([stream[index + self.input_sequence_length:index + self.input_sequence_length + self.prediction_horizon] for index in range(window_count)])
        return inputs, targets

    def _build_model(self, feature_count: int) -> None:
        self.model = SequencePredictorNetwork(feature_count, self.hidden_dim, self.num_layers, self.prediction_horizon, self.config["dropout"])

    def _scaled_windows(self, stream: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inputs, targets = self.create_windows(stream)
        scaled_inputs = self.scaler.transform(inputs.reshape(-1, inputs.shape[-1])).reshape(inputs.shape)
        scaled_targets = self.scaler.transform(targets.reshape(-1, targets.shape[-1])).reshape(targets.shape)
        return scaled_inputs, scaled_targets

    def _predict_scaled(self, inputs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("The LSTM sequence predictor is not trained.")
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(np.asarray(inputs, dtype=np.float32))).cpu().numpy()

    def _window_errors(self, stream: np.ndarray) -> np.ndarray:
        inputs, targets = self._scaled_windows(stream)
        predictions = self._predict_scaled(inputs)
        return np.mean((targets - predictions) ** 2, axis=(1, 2))

    def train(self, normal_train: np.ndarray, normal_validation: np.ndarray) -> Dict:
        """Fit only on normal train windows and derive threshold from normal validation."""
        torch.manual_seed(self.config["random_state"])
        torch.set_num_threads(self.config["num_threads"])
        np.random.seed(self.config["random_state"])
        normal_train = np.asarray(normal_train, dtype=np.float32)
        normal_validation = np.asarray(normal_validation, dtype=np.float32)
        self.feature_count = normal_train.shape[1]
        self.scaler.fit(normal_train)
        inputs, targets = self._scaled_windows(normal_train)
        self._build_model(self.feature_count)
        dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True,
                    num_workers=self.config["num_workers"])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"],
                         weight_decay=self.config["weight_decay"])
        loss_fn = nn.MSELoss()
        for _ in range(self.config["epochs"]):
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(batch_inputs), batch_targets)
                loss.backward()
                optimizer.step()
        validation_errors = self._window_errors(normal_validation)
        self.threshold = float(np.percentile(validation_errors, self.config["threshold_percentile"]))
        self.is_trained = True
        result = {
            "input_sequence_length": self.input_sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "n_training_windows": len(inputs),
            "n_validation_windows": len(validation_errors),
            "threshold": self.threshold,
            "normal_validation_mean_error": float(validation_errors.mean())
        }
        logger.success(f"LSTM next-sequence predictor trained. Threshold: {self.threshold:.6f}")
        return result

    def predict_next_sequence(self, history: np.ndarray) -> np.ndarray:
        """Predict the future horizon after the supplied historical window."""
        if not self.is_trained or self.feature_count is None:
            raise RuntimeError("The LSTM sequence predictor is not trained.")
        history = np.asarray(history, dtype=np.float32)
        if history.shape != (self.input_sequence_length, self.feature_count):
            raise ValueError(f"Expected history shape {(self.input_sequence_length, self.feature_count)}")
        scaled = self.scaler.transform(history).reshape(1, self.input_sequence_length, self.feature_count)
        prediction = self._predict_scaled(scaled)[0]
        return self.scaler.inverse_transform(prediction)

    def score_prediction(self, prediction: np.ndarray, actual: np.ndarray) -> Dict:
        """Compare a completed future prediction with the actual future observations."""
        prediction = np.asarray(prediction, dtype=np.float32)
        actual = np.asarray(actual, dtype=np.float32)
        if prediction.shape != actual.shape:
            raise ValueError("Prediction and actual future sequences must have the same shape.")
        scaled_prediction = self.scaler.transform(prediction)
        scaled_actual = self.scaler.transform(actual)
        error = float(np.mean((scaled_actual - scaled_prediction) ** 2))
        return {"ready": True, "prediction_error": error, "anomaly_threshold": float(self.threshold),
                "is_anomaly": bool(error > self.threshold), "predicted_sequence": prediction, "actual_sequence": actual}

    def observe(self, reading: np.ndarray) -> Dict:
        """Consume one reading and score only when a pending future is complete."""
        reading = np.asarray(reading, dtype=np.float32)
        self.history.append(reading)
        result = {"ready": False, "is_anomaly": False, "prediction_error": None,
                  "anomaly_threshold": self.threshold, "predicted_sequence": None, "actual_sequence": None}
        if self.pending_prediction is not None:
            self.pending_actual.append(reading.copy())
            if len(self.pending_actual) == self.prediction_horizon:
                result = self.score_prediction(self.pending_prediction, np.asarray(self.pending_actual))
                self.pending_prediction = None
                self.pending_actual = []
        if len(self.history) == self.input_sequence_length and self.pending_prediction is None:
            self.pending_prediction = self.predict_next_sequence(np.asarray(self.history))
        return result

    def reset_stream(self) -> None:
        """Clear rolling inference state before starting a new simulated stream."""
        self.history.clear()
        self.pending_prediction = None
        self.pending_actual = []

    def evaluate(self, test_stream: np.ndarray, point_labels: np.ndarray) -> Dict:
        """Evaluate future-window errors using labels only after training is complete."""
        inputs, _ = self.create_windows(test_stream)
        errors = self._window_errors(test_stream)
        labels = np.asarray([int(np.any(point_labels[index + self.input_sequence_length:index + self.input_sequence_length + self.prediction_horizon] != 0)) for index in range(len(inputs))])
        predictions = (errors > self.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        return {"threshold": float(self.threshold),
            "normal_windows": int(np.sum(labels == 0)),
            "anomaly_windows": int(np.sum(labels == 1)),
                "normal_mean_error": float(errors[labels == 0].mean()) if np.any(labels == 0) else 0.0,
                "anomaly_mean_error": float(errors[labels == 1].mean()) if np.any(labels == 1) else 0.0,
                "accuracy": float(accuracy_score(labels, predictions)),
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}

    def save(self) -> None:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save an untrained predictor.")
        torch.save({"state_dict": self.model.state_dict(), "feature_count": self.feature_count,
                    "config": self.config, "threshold": self.threshold}, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load(self) -> bool:
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            return False
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        self.feature_count = int(checkpoint["feature_count"])
        self.threshold = float(checkpoint["threshold"])
        self._build_model(self.feature_count)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.scaler = joblib.load(self.scaler_path)
        self.is_trained = True
        return True