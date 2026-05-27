from ultralytics import YOLO
from collections import deque


class PersonTracker:
    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence_threshold=0.4,
        tracker_config="bytetrack.yaml",
        max_history=30
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0
        self.tracker_config = tracker_config

        self.max_history = max_history
        self.track_history = {}

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    def track(self, frame, frame_id):
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            verbose=False
        )

        tracked_persons = []
        active_track_ids = set()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id != self.person_class_id:
                    continue

                if conf < self.confidence_threshold:
                    continue

                if box.id is None:
                    continue

                track_id = int(box.id[0])
                active_track_ids.add(track_id)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = [x1, y1, x2, y2]
                center = self._get_center(bbox)

                if track_id not in self.track_history:
                    self.track_history[track_id] = deque(maxlen=self.max_history)

                self.track_history[track_id].append({
                    "frame_id": frame_id,
                    "bbox": bbox,
                    "center": center
                })

                tracked_persons.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "center": center,
                    "confidence": conf,
                    "frame_id": frame_id,
                    "history": list(self.track_history[track_id])
                })

        self._cleanup_inactive_tracks(active_track_ids)

        return tracked_persons

    def _cleanup_inactive_tracks(self, active_track_ids):
        stale_ids = set(self.track_history.keys()) - active_track_ids
        for track_id in stale_ids:
            del self.track_history[track_id]
