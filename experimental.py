"""
Idea:

0. For given RECORDING_ID and RECORDING_DIR path to the recording directory:
1. Read MAPPER_DIR/face_detections.csv and choose only when row recording_id == RECORDING_ID
2.


"""

import pandas as pd
import cv2
import os
from collections import defaultdict


REC_ID = "5375387b-6cf7-492e-a866-be923395c692"
REC_DIR = "/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/PAN_test_extraction/2025-06-16_14-22-24-5375387b"

REC_TIMESTAMPS = f"{REC_DIR}/world_timestamps.csv"
REC_SECTIONS = "/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/PAN_test_extraction/sections.csv"
REC_MP4 = "/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/PAN_test_extraction/2025-06-16_14-22-24-5375387b/f62909f8_0.0-493.903.mp4"


MAPPER_DIR = "/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/Sit&Face_FACE-MAPPER_Faces_Manipulative"

MAPPER_CSV = f"{MAPPER_DIR}/face_detections.csv"
MAPPER_SECTIONS = f"{MAPPER_DIR}/sections.csv"

EXTRACTION_DIR = "/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/extraced_faces"


### MISC

def get_vid_data(csv_path: str):
    df = pd.read_csv(csv_path)

    if "timestamp [ns]" not in df.columns:
        raise ValueError("CSV does not contain required column 'timestamp [ns]'")

    start = df["timestamp [ns]"].iloc[0]
    second = df["timestamp [ns]"].iloc[1]
    end = df["timestamp [ns]"].iloc[-1]

    duration_sec = (end - start) / 1e9
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_str = f"{minutes}:{seconds:02d}"

    samp_freq = 1 / ((second-start)/1e9)

    print("Length in seconds:", seconds)
    print("Length in mm:ss:", duration_str)
    print("Sampling rate:", samp_freq, "[Hz]")

    return duration_sec, duration_str, samp_freq


### FACE EXTRACTIONS


def _load_and_filter_csv(csv_path: str, recording_id: str) -> pd.DataFrame:
    """
    Load CSV, validate columns, and filter rows by recording_id.
    """
    print("Opening CSV file...")

    df = pd.read_csv(csv_path)

    required_cols = [
        "recording id",
        "timestamp [ns]",
        "p1 x [px]", "p1 y [px]",
        "p2 x [px]", "p2 y [px]"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")

    # Filter only for the selected recording id
    df = df[df["recording id"] == recording_id].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No rows found for recording id '{recording_id}'")

    print(f"✅ CSV loaded and filtered: {len(df)} rows for recording id {recording_id}")
    df = df.dropna(subset=["p1 x [px]", "p1 y [px]", "p2 x [px]", "p2 y [px]"]).reset_index(drop=True)

    return df

def _get_section_timestamps(csv_path: str, recording_id: str):
    """
    Given a sections CSV and a recording_id, return start and end timestamps (ns)
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ["recording id", "section start time [ns]", "section end time [ns]"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")

    # Filter by recording_id
    df_rec = df[df["recording id"] == recording_id]

    if df_rec.empty:
        raise ValueError(f"No rows found for recording id '{recording_id}'")

    # If multiple rows exist, take first one (or you can return a list)
    start_ts = df_rec["section start time [ns]"].iloc[0]
    end_ts = df_rec["section end time [ns]"].iloc[0]

    return start_ts, end_ts

def extract_frames(video_path: str, csv_path: str, out_dir: str):
    # Load and filter CSV
    df = _load_and_filter_csv(csv_path, recording_id=REC_ID)

    # Drop NaN bbox rows just in case
    df = df.dropna(subset=["p1 x [px]", "p1 y [px]", "p2 x [px]", "p2 y [px]"]).reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save filtered CSV for reference
    csv_out_path = os.path.join(out_dir, "face_frames.csv")
    df.to_csv(csv_out_path, index=False)
    print(f"✅ Filtered CSV saved at {csv_out_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video {video_path}")
    print("🎥 MP4 file opened...")

    fps = cap.get(cv2.CAP_PROP_FPS)
    first_ts, last_ts = _get_section_timestamps(csv_path=REC_SECTIONS, recording_id=REC_ID)

    counter = defaultdict(int)
    total = len(df)

    for i, row in df.iterrows():
        ts = row["timestamp [ns]"]

        # bounding box
        p1x, p1y, p2x, p2y = map(int, [
            row["p1 x [px]"], row["p1 y [px]"],
            row["p2 x [px]"], row["p2 y [px]"]
        ])

        # convert timestamp to frame index
        time_offset = (ts - first_ts) / 1e9
        frame_idx = int(round(time_offset * fps))

        # grab frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"⚠️ Could not read frame {frame_idx} (timestamp {ts})")
            continue

        # clip bbox to frame boundaries
        height, width = frame.shape[:2]
        x1, y1 = max(0, p1x), max(0, p1y)
        x2, y2 = min(width, p2x), min(height, p2y)

        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Invalid bbox at ts={ts}: ({p1x},{p1y})→({p2x},{p2y})")
            continue

        cropped = frame[y1:y2, x1:x2]

        # duplicate handling
        counter[ts] += 1
        suffix = f"_{counter[ts]}" if counter[ts] > 1 else ""
        filename = os.path.join(out_dir, f"{ts}{suffix}.jpg")

        # save crop
        cv2.imwrite(filename, cropped)

        # progress print
        if (i + 1) % 150 == 0 or (i + 1) == total:
            percent = (i + 1) / total * 100
            print(f"Progress: {i+1}/{total} ({percent:.2f}%)")

    cap.release()
    print(f"✅ Done! Cropped faces saved in {out_dir}")


def __main__():
    extract_frames(video_path=REC_MP4,
                   csv_path=MAPPER_CSV,
                   out_dir=EXTRACTION_DIR)

def __test__():
    seconds, mmss, samp_freq = get_vid_data(REC_TIMESTAMPS)


__main__()