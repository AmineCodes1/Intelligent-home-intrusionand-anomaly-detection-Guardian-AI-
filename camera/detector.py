"""Camera-based intrusion detection using YOLOv8 model."""
import numpy as np
import cv2
import os
import time
from typing import List, Dict, Tuple
try:
    from ultralytics import YOLO
except ImportError:
    # Fallback simulation if ultralytics is not installed
    class YOLO:
        def __init__(self, model_path=None):
            self.names = {0: "person"}
        def __call__(self, frame, conf=0.5, verbose=False):
            class Box:
                def __init__(self, cls, conf, xyxy):
                    self.cls = [cls]
                    self.conf = [conf]
                    self.xyxy = [xyxy]
            class Result:
                def __init__(self, boxes):
                    self.boxes = boxes
            # Return empty or dummy result
            return []
from camera.tracker import DeepSortTracker
from camera.video_simulator import VideoSimulator
from utils.config import CAMERA_CONFIG
from utils.logger import logger


class YOLODetector:
    """YOLOv8 detector for human detection."""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.classes = ["person", "car", "dog", "cat", "chair"]
        try:
            # Attempt to load actual YOLOv8 model if available
            self.model = YOLO("yolov8n.pt")
            logger.info("Loaded YOLOv8n model successfully")
        except Exception as e:
            logger.warning(f"Could not load YOLOv8 model, falling back to simulation: {e}")
            self.model = None
        
    def detect(self, frame: np.ndarray, provided_detections: List[Dict] = None) -> List[Dict]:
        """Perform YOLO detection on a frame."""
        if provided_detections is not None:
            return [d for d in provided_detections 
                   if d.get("confidence", 0) >= self.confidence_threshold]
        
        if self.model:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    if label in self.classes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            "class": label,
                            "confidence": float(box.conf[0]),
                            "bbox": (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        })
            return detections

        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.std(gray) > 30:
            detections.append({
                "class": "person",
                "confidence": np.random.uniform(0.6, 0.95),
                "bbox": (
                    np.random.randint(50, frame.shape[1] - 100),
                    np.random.randint(50, frame.shape[0] - 150),
                    np.random.randint(30, 60),
                    np.random.randint(80, 150)
                )
            })
        
        return [d for d in detections if d["confidence"] >= self.confidence_threshold]


class CameraDetector:
    """Camera-based intrusion detection system."""
    
    def __init__(self):
        self.config = CAMERA_CONFIG
        self.detector = YOLODetector(confidence_threshold=self.config["confidence_threshold"])
        self.tracker = DeepSortTracker() if self.config["tracking_enabled"] else None
        self.video_simulator = VideoSimulator(
            width=self.config["resolution"][0],
            height=self.config["resolution"][1],
            fps=self.config["fps"]
        )
        
        self.recording = False
        self.recording_buffer = []
        self.intrusion_count = 0
        
        os.makedirs("outputs/recordings", exist_ok=True)
    
    def process_frame(self, frame: np.ndarray, simulated_detections: List[Dict] = None) -> Dict:
        """Process a single frame for intrusion detection."""
        detections = self.detector.detect(frame, simulated_detections)
        
        person_detections = [d for d in detections if d["class"] == "person"]
        
        tracked_persons = []
        if self.tracker:
            tracked_persons = self.tracker.update(person_detections, frame)
        
        is_intrusion = len(person_detections) > 0
        
        result = {
            "frame": frame,
            "detections": detections,
            "person_count": len(person_detections),
            "tracked_persons": tracked_persons,
            "is_intrusion": is_intrusion
        }
        
        if is_intrusion and not self.recording:
            self._start_recording()
        
        if self.recording:
            # Store frame and result for annotation during video generation
            self.recording_buffer.append({"frame": frame.copy(), "result": result})
            if len(self.recording_buffer) >= self.config["recording_duration"] * self.config["fps"]:
                self._stop_recording()
        
        return result
    
    def _start_recording(self):
        """Start recording video."""
        self.recording = True
        self.recording_buffer = []
        self.intrusion_count += 1
        logger.warning(f"Started recording intrusion event #{self.intrusion_count}")
    
    def _stop_recording(self):
        """Stop recording and save video."""
        self.recording = False
        
        if len(self.recording_buffer) > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/recordings/intrusion_{self.intrusion_count}_{timestamp}.avi"
            
            height, width = self.recording_buffer[0]["frame"].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, self.config["fps"], (width, height))
            
            for frame_data in self.recording_buffer:
                frame = frame_data["frame"]
                result = frame_data["result"]
                # Draw detections and tracking on the frame before writing to video
                annotated_frame = self.draw_detections(frame, result)
                out.write(annotated_frame)
            
            out.release()
            logger.success(f"Saved recording: {filename} ({len(self.recording_buffer)} frames)")
        
        self.recording_buffer = []
    
    def draw_detections(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw detection boxes and tracking info on frame."""
        output = frame.copy()
        
        for det in result["detections"]:
            x, y, w, h = det["bbox"]
            color = (0, 0, 255) if det["class"] == "person" else (0, 255, 0)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(output, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for person in result.get("tracked_persons", []):
            x, y, w, h = person["bbox"]
            # Draw the confirmed DeepSORT track box and ID.
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(output, f"ID: {person['id']}", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        status = "INTRUSION DETECTED" if result["is_intrusion"] else "SECURE"
        status_color = (0, 0, 255) if result["is_intrusion"] else (0, 255, 0)
        cv2.putText(output, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        if self.recording:
            cv2.circle(output, (output.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(output, "REC", (output.shape[1] - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return output
    
    def run_simulation(self, n_frames=100, intrusion_start=30, intrusion_end=60) -> List[Dict]:
        """Run simulation with video frames."""
        logger.info(f"Starting camera simulation: {n_frames} frames")
        
        intrusion_frames = list(range(intrusion_start, intrusion_end))
        results = []
        
        for frame, detections, frame_idx in self.video_simulator.generate_video_stream(
            n_frames=n_frames, intrusion_frames=intrusion_frames):
            
            result = self.process_frame(frame, detections)
            result["frame_idx"] = frame_idx
            results.append(result)
            
            if result["is_intrusion"]:
                logger.alert(f"Frame {frame_idx}: {result['person_count']} person(s) detected!")
        
        if self.recording:
            self._stop_recording()
        
        total_intrusions = sum(1 for r in results if r["is_intrusion"])
        logger.info(f"Simulation complete. Intrusion frames: {total_intrusions}/{n_frames}")
        
        return results
    
    def save_sample_frames(self, results: List[Dict], output_dir="outputs"):
        """Save sample frames from detection results."""
        os.makedirs(output_dir, exist_ok=True)
        
        saved = 0
        for result in results:
            if result["is_intrusion"] and saved < 5:
                frame = self.draw_detections(result["frame"], result)
                filename = f"{output_dir}/detection_frame_{result['frame_idx']}.png"
                cv2.imwrite(filename, frame)
                saved += 1
        
        logger.success(f"Saved {saved} sample detection frames to {output_dir}")


if __name__ == "__main__":
    detector = CameraDetector()
    
    print("\n" + "="*60)
    print("CAMERA-BASED INTRUSION DETECTION SIMULATION")
    print("="*60 + "\n")
    
    results = detector.run_simulation(n_frames=100, intrusion_start=30, intrusion_end=60)
    
    detector.save_sample_frames(results)
    
    print("\n" + "-"*40)
    print("SIMULATION SUMMARY")
    print("-"*40)
    print(f"Total frames processed: {len(results)}")
    print(f"Intrusion detections: {sum(1 for r in results if r['is_intrusion'])}")
    print(f"Recordings saved: {detector.intrusion_count}")
