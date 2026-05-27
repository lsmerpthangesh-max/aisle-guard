from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def estimate(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        poses = []

        for r in results:
            if r.keypoints is None:
                continue

            kpts = r.keypoints.xy.cpu().numpy()
            for person_kpts in kpts:
                poses.append(person_kpts)

        return poses
