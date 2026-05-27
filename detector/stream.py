import cv2
import time


class VideoStream:
    def __init__(self, source, reconnect_delay_sec=2, loop_video=True):
        self.source = source
        self.reconnect_delay_sec = reconnect_delay_sec
        self.loop_video = loop_video
        self.cap = None

    def connect(self):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print("Failed to open video source")
            self.cap = None
            return False

        print("Video source opened")
        return True

    def frames(self):
        while True:
            if self.cap is None:
                if not self.connect():
                    time.sleep(self.reconnect_delay_sec)
                    continue

            ret, frame = self.cap.read()

            if not ret:
                print("Frame read failed")

                self.cap.release()
                self.cap = None

                if self.loop_video:
                    time.sleep(0.5)
                    continue
                else:
                    time.sleep(self.reconnect_delay_sec)
                    continue

            yield frame
