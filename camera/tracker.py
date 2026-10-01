"""DeepSORT multi-object tracking adapter."""
from typing import Dict, List

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    """Use the real DeepSORT tracker while preserving GuardianAI result fields."""

    def __init__(self, max_age=30, n_init=3):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=1.0,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=False
        )

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Update DeepSORT with person detections and return confirmed tracks."""
        raw_detections = [
            (detection["bbox"], detection["confidence"], detection["class"])
            for detection in detections
        ]
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            left, top, right, bottom = track.to_ltrb()
            results.append({
                "id": int(track.track_id),
                "bbox": (
                    int(left),
                    int(top),
                    int(right - left),
                    int(bottom - top)
                ),
                "hits": int(track.hits),
                "age": int(track.age)
            })
        return results

    def reset(self):
        """Reset the underlying DeepSORT tracker."""
        self.tracker.delete_all_tracks()