import os
import json
import cv2
import streamlit as st
from datetime import datetime
from collections import defaultdict
from PIL import Image


EVENTS_DIR = "events"


def load_events():
    events = []

    if not os.path.exists(EVENTS_DIR):
        return events

    for event_name in os.listdir(EVENTS_DIR):
        event_path = os.path.join(EVENTS_DIR, event_name)
        if not os.path.isdir(event_path):
            continue

        metadata_path = os.path.join(event_path, "metadata.json")
        video_path = os.path.join(event_path, "clip.mp4")

        if not os.path.exists(metadata_path) or not os.path.exists(video_path):
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        timestamp = datetime.strptime(metadata["time"], "%Y-%m-%d %H:%M:%S")

        events.append({
            "event_dir": event_path,
            "video": video_path,
            "metadata": metadata,
            "timestamp": timestamp
        })

    events.sort(key=lambda x: x["timestamp"], reverse=True)
    return events


def extract_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def group_events_by_date(events):
    grouped = defaultdict(list)
    for event in events:
        date_key = event["timestamp"].strftime("%A, %d %B %Y")
        grouped[date_key].append(event)
    return grouped


st.set_page_config(
    page_title="Aisle Guard – Incident History",
    layout="wide"
)

st.title("History")

events = load_events()
grouped_events = group_events_by_date(events)

if not events:
    st.info("No shoplifting incidents detected yet.")
    st.stop()

for date, day_events in grouped_events.items():
    st.subheader(date)

    for event in day_events:
        col1, col2 = st.columns([1, 5])

        thumbnail = extract_thumbnail(event["video"])

        with col1:
            if thumbnail:
                st.image(thumbnail, width=120)

        with col2:
            meta = event["metadata"]
            time_str = event["timestamp"].strftime("%I:%M %p")

            st.markdown(f"**Shoplifting Detected**")
            st.caption(f"{time_str} · Camera {meta['camera_id']}")

            with st.expander("View details"):
                st.video(event["video"])
                st.json(meta)

        st.divider()
