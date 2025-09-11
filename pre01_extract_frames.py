import pandas as pd
import cv2
import os
from collections import defaultdict

import pre00_rec_path_handling as pre00

from P01_config import MAPPER_CSV
from P01_config import EXTRACTION_DIR

from P02_model_parameters import CNN_INPUT_SIZE, IMAGE_PAD_RATIO


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

def extract_frames(rec_id: str, rec_dict: dict, mapper_detections: str):
    # Load and filter CSV
    df = _load_and_filter_csv(mapper_detections, recording_id=rec_id)

    # Drop NaN bbox rows just in case
    df = df.dropna(subset=["p1 x [px]", "p1 y [px]", "p2 x [px]", "p2 y [px]"]).reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(rec_dict['extract'], exist_ok=True)

    # Save filtered CSV for reference
    csv_out_path = os.path.join(rec_dict['extract'], "face_frames.csv")
    df.to_csv(csv_out_path, index=False)
    print(f"✅ Filtered CSV saved at {csv_out_path}")

    # Open video
    cap = cv2.VideoCapture(rec_dict['video'])
    if not cap.isOpened():
        raise IOError(f"Could not open video {rec_dict['video']}")
    print("🎥 MP4 file opened...")

    fps = cap.get(cv2.CAP_PROP_FPS)
    first_ts, last_ts = _get_section_timestamps(csv_path=rec_dict['sections'], recording_id=rec_id)

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

        height, width = frame.shape[:2]

        # add padding
        w = p2x - p1x
        h = p2y - p1y
        pad = int(max(w, h) * IMAGE_PAD_RATIO)

        x1 = max(0, p1x - pad)
        y1 = max(0, p1y - pad)
        x2 = min(width, p2x + pad)
        y2 = min(height, p2y + pad)

        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Invalid bbox at ts={ts}: ({p1x},{p1y})→({p2x},{p2y})")
            continue

        cropped = frame[y1:y2, x1:x2]

        # resize without warping (letterbox)
        ch, cw = cropped.shape[:2]
        scale = CNN_INPUT_SIZE / max(ch, cw)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # pad to square
        top = (CNN_INPUT_SIZE - new_h) // 2
        bottom = CNN_INPUT_SIZE - new_h - top
        left = (CNN_INPUT_SIZE - new_w) // 2
        right = CNN_INPUT_SIZE - new_w - left

        squared = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])  # black padding

        # handle duplicates
        counter[ts] += 1
        suffix = f"_{counter[ts]}" if counter[ts] > 1 else ""
        filename = os.path.join(rec_dict['extract'], f"{ts}{suffix}.jpg")

        # save crop
        cv2.imwrite(filename, squared)

        # progress
        if (i + 1) % 150 == 0 or (i + 1) == total:
            percent = (i + 1) / total * 100
            print(f"Progress: {i + 1}/{total} ({percent:.2f}%)")

    cap.release()
    print(f"✅ Done! Cropped faces saved in {rec_dict['extract']}")


def __main__():
    recs_dict = pre00.fetch_rec_paths()

    os.makedirs(EXTRACTION_DIR, exist_ok=False)
    for rec_id, rec_dict in recs_dict.items():
        extract_frames(rec_id=rec_id,
                       rec_dict=rec_dict,
                       mapper_detections=MAPPER_CSV)

__main__()