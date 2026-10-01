"""End-to-end YOLOv8 + DeepSORT processing benchmark."""
import argparse
import time
from typing import Dict, List

import numpy as np

from camera.detector import CameraDetector
from camera.video_simulator import VideoSimulator
from utils.config import CAMERA_CONFIG


def _synchronize_cuda() -> None:
    """Wait for asynchronous CUDA work when a CUDA-enabled PyTorch exists."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_pipeline(frame_count: int = 300, warmup_frames: int = 30) -> Dict[str, float]:
    """Benchmark the existing detector/tracker loop without rendering or recording."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if warmup_frames < 0 or warmup_frames >= frame_count:
        raise ValueError("warmup_frames must be non-negative and less than frame_count")

    video_simulator = VideoSimulator(
        width=CAMERA_CONFIG["resolution"][0],
        height=CAMERA_CONFIG["resolution"][1],
        fps=CAMERA_CONFIG["fps"]
    )
    frames: List[np.ndarray] = []
    intrusion_frames = range(frame_count // 3, 2 * frame_count // 3)
    for frame, _, _ in video_simulator.generate_video_stream(
        n_frames=frame_count,
        intrusion_frames=list(intrusion_frames)
    ):
        frames.append(frame)

    detector = CameraDetector()
    if not detector.detector.using_real_yolo:
        raise RuntimeError(
            "Real Ultralytics YOLO is unavailable. Install requirements.txt "
            "before running the YOLOv8 + DeepSORT benchmark."
        )

    for frame in frames[:warmup_frames]:
        detector.process_frame(frame, simulated_detections=None, manage_recording=False)

    if detector.tracker:
        detector.tracker.reset()

    _synchronize_cuda()
    start = time.perf_counter()
    detected_person_frames = 0
    for frame in frames[warmup_frames:]:
        result = detector.process_frame(frame, simulated_detections=None, manage_recording=False)
        detected_person_frames += int(result["person_count"] > 0)
    _synchronize_cuda()
    elapsed_seconds = time.perf_counter() - start

    benchmarked_frames = frame_count - warmup_frames
    processing_fps = benchmarked_frames / elapsed_seconds if elapsed_seconds else float("inf")
    return {
        "input_video_fps": float(video_simulator.fps),
        "warmup_frames": warmup_frames,
        "frames_benchmarked": benchmarked_frames,
        "processing_time_seconds": elapsed_seconds,
        "pipeline_fps": processing_fps,
        "average_ms_per_frame": (elapsed_seconds / benchmarked_frames) * 1000,
        "detected_person_frames": detected_person_frames,
        "real_time_capable": processing_fps >= video_simulator.fps
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the YOLOv8 + DeepSORT pipeline")
    parser.add_argument("--frames", type=int, default=300, help="Total simulator frames, including warm-up")
    parser.add_argument("--warmup", type=int, default=30, help="Frames discarded before timing")
    args = parser.parse_args()

    result = benchmark_pipeline(args.frames, args.warmup)
    print(f"Input video FPS: {result['input_video_fps']:.2f}")
    print(f"Warm-up frames: {result['warmup_frames']}")
    print(f"Frames benchmarked: {result['frames_benchmarked']}")
    print(f"Processing time: {result['processing_time_seconds']:.2f} s")
    print(f"Pipeline FPS: {result['pipeline_fps']:.2f}")
    print(f"Average latency: {result['average_ms_per_frame']:.2f} ms/frame")
    print(f"Frames with detected people: {result['detected_person_frames']}")
    print(f"Real-time capable at {result['input_video_fps']:.2f} FPS: {'YES' if result['real_time_capable'] else 'NO'}")


if __name__ == "__main__":
    main()