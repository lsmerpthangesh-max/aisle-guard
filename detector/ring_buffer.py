from collections import deque
import time

class RingBuffer:
    def __init__(self, max_seconds, fps):
        self.max_seconds = max_seconds
        self.fps = fps
        self.max_frames = int(max_seconds * fps)
        self.buffer = deque(maxlen=self.max_frames)

    def add(self, frame):
        self.buffer.append({
            "timestamp": time.time(),
            "frame": frame.copy()
        })

    def get_frames(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()