# aisle-guard

![inspired-by-lexius](https://github.com/user-attachments/assets/74f03b5b-8bb3-46cc-bc5f-7dd0c761ea1e)

Aisle Guard is a lightweight, end-to-end computer vision system for detecting and logging **potential shoplifting events** from retail camera footage.

It works on both **live webcam feeds** and **pre-recorded videos**, performs real-time person tracking and behavior reasoning, and saves short video clips only when a situation is classified as **risky**.

The system is designed to be interpretable, and modular, prioritizing explainable behavior logic.

---

## Capabilities as of now

- Live or video-based camera stream ingestion
- Real-time person detection and tracking
- Temporal behavior modeling using a state machine
- Risk scoring based on shelf interaction and cart placement
- Automatic incident clip saving with metadata
- Human pose visualization for better interpretability
- Minimal dashboard to review incidents

---

## How It Works (System Flow)

The system runs as a continuous loop and is composed of the following stages:

### 1. Video Stream Ingestion
- Input can be:
  - A webcam (OpenCV VideoCapture)
  - A video file (looped or one-shot)
- Implemented in `detector/stream.py`

### 2. Frame Sampling
- Heavy operations are not run on every frame
- A simple sampler processes every Nth frame
- Improves performance and stability
- Implemented in `detector/sampler.py`

### 3. Person Detection and Tracking
- `YOLOv8` is used for person detection
- `ByteTrack` assigns consistent IDs across frames
- Each person maintains a short movement history
- Implemented in `detector/tracker.py`

### 4. Ring Buffer (Pre-Event Memory)
- A rolling in-memory buffer keeps the last 10–15 seconds of frames
- Frames are always added, regardless of detection
- Enables saving context before an event is detected
- Implemented in `detector/ring_buffer.py`

### 5. Behavior Reasoning (State Machine)
Each tracked person moves through explicit states:

- IDLE
- INTERACTING_WITH_SHELF
- CART_CHECK
- MOVING_AWAY
- SAFE
- RISK

Risk accumulates over time based on:
- Shelf dwell duration
- Absence of cart/basket interaction
- Movement away from shelf after interaction

No backpacks or handbags are treated as valid carts.

Implemented in `detector/behavior.py`.

### 6. Event Trigger and Clip Saving
- When a person enters the `RISK` state:
  - The ring buffer is dumped to a video clip
  - Metadata is saved alongside the clip
- Each `person ID` is logged only once per session

Sample of saved structure:
```json
{
  "time": "2026-01-24 11:10:12",
  "camera_id": "cam_01",
  "person_id": 1,
  "risk_score": 0.75
}
```
Implemented in `detector/event_logger.py`

### 7. Pose Estimation (Visualization)
- `yolov8n-pose` model is used to draw body keypoints and skeletons
- Pose inference runs only on sampled frames
- Used purely for visualization and interpretability

Implemented in `detector/pose.py`.

### 8. Incident Review Dashboard
- Streamlit app reads from the `events/` directory
- Incidents are grouped by date and sorted by time
- Each event shows:
  - Thumbnail
  - Timestamp
  - Camera ID
  - Video playback
  - Metadata

Implemented in `dashboard.py`.


---

## In case you want to try this :

1. clone the repo :
   ```bash
   git clone https://github.com/Keerthanareddy17/aisle-guard.git
   ```
2. Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # macOS / Linux
   env\Scripts\activate    # Windows
   ```
3. install the requirements :
   ```bash
   pip install -r requirements.txt
   ```
4. Run Detection Pipeline :
- Edit `main.py` to choose input mode:
  ```python
  MODE = "live"   # webcam
  # or
  MODE = "video"
  VIDEO_PATH = "test_sample.mp4"
  ```
- Then run:
  ```python
  python main.py
  ```


  <img width="1710" height="980" alt="Screenshot 2026-01-24 at 11 30 17" src="https://github.com/user-attachments/assets/d2abe859-b15a-4ba3-8576-a008105e7abd" />
  
- To view the dashboard, run :
  ```python
  streamlit run dashboard.py
  ```

---

## Output

Only risky situations are saved

Each incident includes:
- Short video clip
- Timestamp
- Camera ID
- Person ID
- Risk score
  
This allows fast review without storing unnecessary footage.

<img width="1533" height="771" alt="Screenshot 2026-01-24 at 11 27 38" src="https://github.com/user-attachments/assets/2fe19acc-5cc2-43fa-8c8a-44e4d0d34905" />

---

## Possible Future Improvements

- Interactive shelf/cart zone editor
- Multi-camera support
- Risk cooldown and reset logic
- Event review and annotation in dashboard
- Integration with alerting systems (email, Slack)
- Model fine-tuning for better detection of carts and baskets

---

If you find this project useful or interesting, feel free to star the repository ⭐️, or adapt it for your own experiments. Contributions, ideas, and critical feedback are always welcome.

Thanks for checking it out.

Here's my [LinkedIn](https://www.linkedin.com/in/keerthana-reddy-katasani-b07238268/) ✌️

![developed-by-~kr](https://github.com/user-attachments/assets/793a09ac-eb1f-41f1-bb92-dd7af9f74c63)

