import os
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import cv2

from config import P01_extraction_config as P01
from config import P02_model_config as P02

"""
Idea:
on extraction, put the points and boxes so that you can see.
also what the padding is etc.
"""

def get_vid_data(csv_path: str):
    """
    Compute sampling frequency and duration from timestamps.
    """
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

    samp_freq = 1 / ((second - start) / 1e9)

    print(f"Length in seconds: {int(duration_sec)}")
    print(f"Length in mm:ss: {duration_str}")
    print(f"Sampling rate: {samp_freq:.2f} Hz")

    return duration_sec, duration_str, samp_freq


def _load_and_filter_csv(mapper_detections: str, recording_id: str):
    """
    Load face_detections.csv and filter by recording_id.
    """
    df = pd.read_csv(mapper_detections)
    df = df[df["recording id"] == recording_id].reset_index(drop=True)
    return df


# ======================================================
#  Frame extraction for a single recording
# ======================================================

def extract_frames(rec_dict: dict, mapper_detections: str):
    """
    Extracts all face frames for one recording and saves them to its extraction_dir.
    Adds a 'suffix' column in face_frames.csv. Every saved image filename includes the suffix (even 0).
    """
    recording_id = rec_dict["recording_id"]
    video_path = rec_dict["mp4_path"]
    extract_dir = Path(rec_dict["extraction_dir"])

    # --- Load detections ---
    df = _load_and_filter_csv(mapper_detections, recording_id)
    if df.empty:
        print(f"⚠️ No face detections for {recording_id}")
        return

    df = df.dropna(subset=["p1 x [px]", "p1 y [px]", "p2 x [px]", "p2 y [px]"]).reset_index(drop=True)
    os.makedirs(extract_dir, exist_ok=True)

    # --- Assign suffixes per timestamp ---
    df = df.sort_values("timestamp [ns]").reset_index(drop=True)
    df["suffix"] = 0
    last_ts = None
    counter = -1
    for i, row in df.iterrows():
        ts = int(row["timestamp [ns]"])
        if ts != last_ts:
            counter = 0
            last_ts = ts
        else:
            counter += 1
        df.at[i, "suffix"] = counter

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Opened video {video_path} | FPS: {fps:.2f}")

    rec_start_ns = int(rec_dict["start_time"])
    section_start_ns = rec_dict.get("section_start_time_ns")
    section_end_ns = rec_dict.get("section_end_time_ns")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total = len(df)

    for i, row in df.iterrows():
        ts = int(row["timestamp [ns]"])
        suffix = int(row["suffix"])

        # --- Compute frame index relative to recording start ---
        time_offset = (ts - rec_start_ns) / 1e9  # seconds
        if time_offset < 0:
            print(f"⚠️ Detection {ts} occurs before video start — skipping")
            continue

        frame_idx = int(round(time_offset * fps))
        if frame_idx < 0 or frame_idx >= frame_count:
            print(f"⚠️ Computed frame_idx {frame_idx} outside video frames — skipping")
            continue

        # --- Optional: skip detections outside manipulative section ---
        if section_start_ns and section_end_ns:
            if ts < section_start_ns or ts > section_end_ns:
                continue

        # --- Seek and read frame ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"⚠️ Could not read frame {frame_idx} (timestamp {ts})")
            continue

        # --- Crop bounding box ---
        p1x, p1y, p2x, p2y = map(int, [row["p1 x [px]"], row["p1 y [px]"], row["p2 x [px]"], row["p2 y [px]"]])
        height, width = frame.shape[:2]
        w, h = p2x - p1x, p2y - p1y
        pad = int(max(w, h) * P02.IMAGE_PAD_RATIO)

        x1 = max(0, p1x - pad)
        y1 = max(0, p1y - pad)
        x2 = min(width, p2x + pad)
        y2 = min(height, p2y + pad)
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Invalid bbox at ts={ts}")
            continue

        cropped = frame[y1:y2, x1:x2]

        # --- Resize proportionally ---
        ch, cw = cropped.shape[:2]
        scale = P02.CNN_INPUT_SIZE / max(ch, cw)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # --- Pad to square ---
        top = (P02.CNN_INPUT_SIZE - new_h) // 2
        bottom = P02.CNN_INPUT_SIZE - new_h - top
        left = (P02.CNN_INPUT_SIZE - new_w) // 2
        right = P02.CNN_INPUT_SIZE - new_w - left
        squared = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # --- Save frame with suffix ---
        filename = extract_dir / f"{ts}_{suffix}.jpg"
        cv2.imwrite(str(filename), squared)
        df.at[i, "frame_path"] = str(filename)

        if (i + 1) % 150 == 0 or (i + 1) == total:
            percent = (i + 1) / total * 100
            print(f"Progress: {i + 1}/{total} ({percent:.1f}%)")

    # --- Save filtered detections with suffix ---
    csv_out_path = extract_dir / "face_frames.csv"
    df.to_csv(csv_out_path, index=False)
    print(f"✅ Filtered detections saved at {csv_out_path}")
    cap.release()
    print(f"✅ Done extracting {total} faces for {recording_id}")



# ======================================================
#  Run for all recordings
# ======================================================

def extract_faces_for_all(FACE_MAPPER_DIR: str, recordings_info_path: str, subset_ids: list[str] | None = None):
    """
    Runs extraction for selected recordings (or all, if none specified).
    """
    FACE_MAPPER_DIR = Path(FACE_MAPPER_DIR)
    mapper_detections = FACE_MAPPER_DIR / "face_detections.csv"

    with open(recordings_info_path, "r") as f:
        recordings_info = json.load(f)

    if subset_ids is not None:
        subset_ids = set(subset_ids)
        recordings_info = {k: v for k, v in recordings_info.items() if k in subset_ids}
        print(f"Processing only {len(recordings_info)} recordings (subset mode).")
    else:
        print(f"Processing all {len(recordings_info)} recordings.")

    for rec_id, rec_data in recordings_info.items():
        print("\n" + "="*70)
        print(f"▶ Processing recording {rec_id} ({rec_data['rec_name']})")

        # --- Check sampling frequency ---
        _, _, freq = get_vid_data(rec_data["timestamps_path"])
        assert round(freq) == 20, f"⚠️ Sampling rate mismatch for {rec_id}: {freq:.2f} Hz"

        # --- Add recording_id for convenience ---
        rec_data["recording_id"] = rec_id

        # --- Extract frames ---
        extract_frames(rec_data, str(mapper_detections))

    print("\n✅ Extraction complete!")
