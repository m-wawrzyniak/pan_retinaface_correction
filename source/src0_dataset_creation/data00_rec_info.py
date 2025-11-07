import os
import json
import pandas as pd
from pathlib import Path


def build_recordings_info(timeseries_data_dir: str, sections_csv_path: str):
    """
    Builds a dictionary with metadata and file paths for each recording in the given directory.
    Creates an 'extracted_frames' folder inside each recording directory.
    Extracts section start/end times from a given sections.csv for manipulative.begin events.
    Saves the resulting dictionary as 'recordings_info.json' in the same folder.

    Parameters
    ----------
    timeseries_data_dir : str
        Path to the base folder containing the per-recording directories.
    sections_csv_path : str
        Path to the sections.csv file (e.g. from FACE_MAPPER directory).
    """
    timeseries_data_dir = Path(timeseries_data_dir)
    sections_csv_path = Path(sections_csv_path)
    recordings_info = {}

    assert sections_csv_path.exists(), f"❌ sections.csv not found at {sections_csv_path}"

    # --- Load and filter sections.csv ---
    sections_df = pd.read_csv(sections_csv_path)
    manipulative_sections = sections_df[sections_df["start event name"] == "manipulative.begin"]

    for recording_dir in timeseries_data_dir.iterdir():
        if not recording_dir.is_dir():
            continue

        info_path = recording_dir / "info.json"
        timestamps_path = recording_dir / "world_timestamps.csv"
        mp4_files = list(recording_dir.glob("*.mp4"))

        # --- Sanity checks ---
        if not info_path.exists():
            print(f"⚠️ Skipping {recording_dir.name}: missing info.json")
            continue
        if not timestamps_path.exists():
            print(f"⚠️ Skipping {recording_dir.name}: missing world_timestamps.csv")
            continue
        if len(mp4_files) != 1:
            print(f"⚠️ Skipping {recording_dir.name}: expected 1 mp4, found {len(mp4_files)}")
            continue

        # --- Parse info.json ---
        with open(info_path, "r") as f:
            info = json.load(f)

        recording_id = info.get("recording_id")
        recording_name = info.get("template_data", {}).get("recording_name")
        start_time = info.get("start_time")
        gaze_freq = info.get("gaze_frequency")

        # --- Assertions ---
        assert gaze_freq == 200, f"Gaze frequency not 200 in {recording_dir.name}"

        # --- Create extracted_frames directory ---
        extraction_dir = recording_dir / "extracted_frames"
        extraction_dir.mkdir(exist_ok=True)

        # --- Get section start/stop times for manipulative.begin ---
        section_rows = manipulative_sections[manipulative_sections["recording id"] == recording_id]
        if section_rows.empty:
            print(f"⚠️ No manipulative.begin section found for {recording_id}")
            section_start, section_end = None, None
        else:
            # take first if multiple found
            row = section_rows.iloc[0]
            section_start = int(row["section start time [ns]"])
            section_end = int(row["section end time [ns]"])

        # --- Build entry ---
        recordings_info[recording_id] = {
            "rec_name": recording_name,
            "start_time": start_time,
            "mp4_path": str(mp4_files[0]),
            "timestamps_path": str(timestamps_path),
            "extraction_dir": str(extraction_dir),
            "section_start_time_ns": section_start,
            "section_end_time_ns": section_end
        }

    # --- Save the results ---
    out_path = timeseries_data_dir / "recordings_info.json"
    with open(out_path, "w") as f:
        json.dump(recordings_info, f, indent=4)

    print(f"✅ Saved recordings_info.json with {len(recordings_info)} recordings at: {out_path}")
    return recordings_info


# Example usage:
if __name__ == "__main__":
    TIMESERIES_DATA = "/home/mateusz-wawrzyniak/Desktop/IP_PAN_Videos/Timeseries Data + Scene Video/"
    SECTIONS_CSV = "/home/mateusz-wawrzyniak/Desktop/IP_PAN_Videos/Sit&Face_FACE-MAPPER_Faces_Manipulative/sections.csv"

    build_recordings_info(TIMESERIES_DATA, SECTIONS_CSV)
