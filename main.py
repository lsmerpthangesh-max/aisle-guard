import cv2
from detector.stream import VideoStream
from detector.sampler import FrameSampler
from detector.tracker import PersonTracker
from detector.ring_buffer import RingBuffer
from detector.behavior import BehaviorEngine
from detector.event_logger import EventLogger
from detector.pose import PoseEstimator


MODE = "video"

VIDEO_PATH = "test_sample.mp4"

FPS = 30
BUFFER_SECONDS = 15
PROCESS_EVERY_N_FRAMES = 5

SHELF_ZONE = {
    "x1": 100,
    "y1": 100,
    "x2": 500,
    "y2": 400
}

CART_ZONE = {
    "x1": 520,
    "y1": 300,
    "x2": 800,
    "y2": 600
}


def point_in_zone(cx, cy, zone):
    return (
        zone["x1"] <= cx <= zone["x2"] and
        zone["y1"] <= cy <= zone["y2"]
    )


def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        state = track.get("state", "SAFE")
        risk = track.get("risk", 0.0)

        if state == "RISK":
            color = (0, 0, 255)
            label = f"RISK {risk:.2f}"
            font_scale = 1.3
            thickness = 3
        else:
            color = (0, 255, 0)
            label = "SAFE"
            font_scale = 0.9
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness
        )


def print_debug_tracks(tracks):
    print("\n--- TRACK DEBUG ---")
    for track in tracks:
        track_id = track["track_id"]
        center = track["center"]
        history = track["history"]

        print(f"Person ID: {track_id}")
        print(f"  Current center: {center}")
        print(f"  History length: {len(history)}")

        for h in history[-5:]:
            print(f"    Frame {h['frame_id']} -> Center {h['center']}")


POSE_CONNECTIONS = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def draw_pose(frame, keypoints):
    for idx, (x, y) in enumerate(keypoints):
        if idx in [0, 1, 2, 3, 4]:
            color = (255, 255, 0)
        elif idx in [5, 6, 7, 8, 9, 10]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    for i, j in POSE_CONNECTIONS:
        x1, y1 = keypoints[i]
        x2, y2 = keypoints[j]
        cv2.line(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 255),
            2
        )


def main():
    if MODE == "video":
        source = VIDEO_PATH
        loop_video = True
    else:
        source = 0
        loop_video = False

    stream = VideoStream(source=source, loop_video=loop_video)

    ring_buffer = RingBuffer(
        max_seconds=BUFFER_SECONDS,
        fps=FPS
    )

    sampler = FrameSampler(process_every_n_frames=PROCESS_EVERY_N_FRAMES)
    tracker = PersonTracker(model_path="yolov8n.pt")
    behavior = BehaviorEngine(fps=FPS)

    event_logger = EventLogger(
        base_dir="events",
        fps=FPS,
        camera_id="cam_01"
    )

    pose_estimator = PoseEstimator()
    ENABLE_POSE = True

    frame_id = 0
    last_tracks = []
    last_poses = []

    for frame in stream.frames():
        frame_id += 1
        ring_buffer.add(frame)

        if frame_id % 100 == 0:
            print(
                f"Ring buffer frames: "
                f"{len(ring_buffer.get_frames())} / {ring_buffer.max_frames}"
            )

        if sampler.should_process():
            last_tracks = tracker.track(frame, frame_id)
            print_debug_tracks(last_tracks)

            last_poses = []
            if ENABLE_POSE:
                last_poses = pose_estimator.estimate(frame)

            for track in last_tracks:
                x1, y1, x2, y2 = track["bbox"]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                in_shelf = point_in_zone(cx, cy, SHELF_ZONE)
                in_cart = point_in_zone(cx, cy, CART_ZONE)

                state, risk = behavior.update(
                    track_id=track["track_id"],
                    in_shelf_zone=in_shelf,
                    in_cart_zone=in_cart,
                    frame_id=frame_id
                )

                track["state"] = state
                track["risk"] = risk

                if state == "RISK":
                    event_logger.log_event(
                        ring_buffer=ring_buffer,
                        track_id=track["track_id"],
                        risk_score=risk
                    )

        draw_tracks(frame, last_tracks)

        if ENABLE_POSE:
            for pose in last_poses:
                draw_pose(frame, pose)

        cv2.imshow("Aisle Guard", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
