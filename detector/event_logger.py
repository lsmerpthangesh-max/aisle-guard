import os
import json
import cv2
from datetime import datetime


class EventLogger:
    def __init__(self, base_dir="events", fps=30, camera_id="cam_01"):
        self.base_dir = base_dir
        self.fps = fps
        self.camera_id = camera_id
        self.logged_events = set()

        os.makedirs(self.base_dir, exist_ok=True)

    def log_event(self, ring_buffer, track_id, risk_score):
        if track_id in self.logged_events:
            return

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        event_dir = os.path.join(self.base_dir, f"event_{timestamp}")
        os.makedirs(event_dir, exist_ok=True)

        clip_path = os.path.join(event_dir, "clip.mp4")
        metadata_path = os.path.join(event_dir, "metadata.json")

        frames = ring_buffer.get_frames()
        if not frames:
            return

        height, width, _ = frames[0]["frame"].shape
        writer = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height)
        )

        for item in frames:
            writer.write(item["frame"])

        writer.release()

        metadata = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": self.camera_id,
            "person_id": track_id,
            "risk_score": round(float(risk_score), 3)
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logged_events.add(track_id)
