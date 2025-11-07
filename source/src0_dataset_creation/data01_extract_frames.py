FACE_MAPPER_DIR = "/home/mateusz-wawrzyniak/Desktop/IP_PAN_Videos/Sit&Face_FACE-MAPPER_Faces_Manipulative/"
RECORDINGS_INFO_PATH = "/home/mateusz-wawrzyniak/Desktop/IP_PAN_Videos/Timeseries Data + Scene Video/recordings_info.json"
REC_SUBSET = [
    "dcd95915-e5b0-4220-99b2-19c883d41d33"
]
import os
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import cv2

# --- Constants ---
IMAGE_PAD_RATIO = 0.25    # add 25% padding around bbox
CNN_INPUT_SIZE = 224      # output image size for CNN


# ======================================================
#  Utility function
# ======================================================

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
    Uses section_start_time_ns and section_end_time_ns from rec_dict.
    """
    recording_id = rec_dict["recording_id"]
    video_path = rec_dict["mp4_path"]
    extract_dir = rec_dict["extraction_dir"]

    # --- Load detections ---
    df = _load_and_filter_csv(mapper_detections, recording_id)
    if df.empty:
        print(f"⚠️ No face detections for {recording_id}")
        return

    df = df.dropna(subset=["p1 x [px]", "p1 y [px]", "p2 x [px]", "p2 y [px]"]).reset_index(drop=True)
    os.makedirs(extract_dir, exist_ok=True)

    # --- Save filtered detections for reference ---
    csv_out_path = os.path.join(extract_dir, "face_frames.csv")
    df.to_csv(csv_out_path, index=False)
    print(f"✅ Filtered detections saved at {csv_out_path}")

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Opened video {video_path} | FPS: {fps:.2f}")

    # --- Reference timestamp: start of recording ---
    rec_start_ns = int(rec_dict["start_time"])
    section_start_ns = rec_dict.get("section_start_time_ns")
    section_end_ns = rec_dict.get("section_end_time_ns")
    if rec_start_ns is None:
        print(f"⚠️ Missing recording start_time for {recording_id}")
        return

    # --- Video properties ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Could not read FPS from video {video_path}")
    print(f"🎥 Opened video {video_path} | FPS: {fps:.2f}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    counter = defaultdict(int)
    total = len(df)

    for i, row in df.iterrows():
        ts = int(row["timestamp [ns]"])  # absolute timestamp (ns)

        # --- Compute frame index relative to recording start ---
        time_offset = (ts - rec_start_ns) / 1e9  # seconds since video start

        if time_offset < 0:
            print(f"⚠️ Detection {ts} occurs before video start ({rec_start_ns}) — skipping")
            continue

        frame_idx = int(round(time_offset * fps))

        # --- Bound check ---
        if frame_idx < 0 or frame_idx >= frame_count:
            print(f"⚠️ Computed frame_idx {frame_idx} outside video frames [0, {frame_count-1}] — skipping")
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

        # --- Extract bounding box ---
        p1x, p1y, p2x, p2y = map(int, [
            row["p1 x [px]"], row["p1 y [px]"],
            row["p2 x [px]"], row["p2 y [px]"]
        ])

        height, width = frame.shape[:2]
        w, h = p2x - p1x, p2y - p1y
        pad = int(max(w, h) * IMAGE_PAD_RATIO)

        x1 = max(0, p1x - pad)
        y1 = max(0, p1y - pad)
        x2 = min(width, p2x + pad)
        y2 = min(height, p2y + pad)

        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Invalid bbox at ts={ts}: ({p1x},{p1y})→({p2x},{p2y})")
            continue

        cropped = frame[y1:y2, x1:x2]

        # --- Resize proportionally to CNN_INPUT_SIZE ---
        ch, cw = cropped.shape[:2]
        scale = CNN_INPUT_SIZE / max(ch, cw)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # --- Pad to square ---
        top = (CNN_INPUT_SIZE - new_h) // 2
        bottom = CNN_INPUT_SIZE - new_h - top
        left = (CNN_INPUT_SIZE - new_w) // 2
        right = CNN_INPUT_SIZE - new_w - left
        squared = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # --- Save frame ---
        counter[ts] += 1
        suffix = f"_{counter[ts]}" if counter[ts] > 1 else ""
        filename = os.path.join(extract_dir, f"{ts}{suffix}.jpg")
        cv2.imwrite(filename, squared)

        if (i + 1) % 150 == 0 or (i + 1) == total:
            percent = (i + 1) / total * 100
            print(f"Progress: {i + 1}/{total} ({percent:.1f}%)")

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


if __name__ == "__main__":
    extract_faces_for_all(FACE_MAPPER_DIR, RECORDINGS_INFO_PATH, subset_ids=REC_SUBSET)