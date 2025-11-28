import os
import json
import pandas as pd
from pathlib import Path

from config import P01_extraction_config as P01

def build_recordings_info(
        timeseries_data_dir: str,
        sections_csv_path: str,
        section_name: str,
        extraction_root: str,
        manual_csv_root: str,
        model_csv_root: str,
        json_path: str,
        rec_subset: list
    ):
    """
    Builds a dictionary with metadata and file paths for each recording in the given directory.
    For each recording, creates an EMPTY directory at out_path/<recording_name>/,
    used for storing extracted frames (so original data is not modified).

    Saves the final dictionary as recordings_info.json inside out_path.

    Parameters
    ----------
    timeseries_data_dir : str
        Path to the base folder containing recording directories.
    sections_csv_path : str
        Path to sections.csv file.
    section_name : str
        Section name to filter (e.g. 'manipulative.begin').
    json_path : str
        Base directory where per-recording extraction dirs are created.
    """

    timeseries_data_dir = Path(timeseries_data_dir)
    sections_csv_path = Path(sections_csv_path)

    assert sections_csv_path.exists(), f"❌ sections.csv not found at {sections_csv_path}"

    # --- Load sections ---
    sections_df = pd.read_csv(sections_csv_path)
    target_sections = sections_df[sections_df["start event name"] == section_name]

    recordings_info = {}

    for recording_dir in timeseries_data_dir.iterdir():
        if not recording_dir.is_dir():
            continue

        info_path = recording_dir / "info.json"
        timestamps_path = recording_dir / "world_timestamps.csv"
        mp4_files = list(recording_dir.glob("*.mp4"))

        # Basic checks
        if not info_path.exists():
            print(f"⚠️ Skipping {recording_dir.name}: missing info.json")
            continue
        if not timestamps_path.exists():
            print(f"⚠️ Skipping {recording_dir.name}: missing world_timestamps.csv")
            continue
        if len(mp4_files) != 1:
            print(f"⚠️ Skipping {recording_dir.name}: expected 1 mp4, found {len(mp4_files)}")
            continue

        # --- Load recording info ---
        with open(info_path, "r") as f:
            info = json.load(f)

        recording_id = info.get("recording_id")
        if not recording_id in rec_subset:
            continue
        recording_name = info.get("template_data", {}).get("recording_name")
        start_time = info.get("start_time")
        gaze_freq = info.get("gaze_frequency")

        assert gaze_freq == 200, f"Gaze frequency not 200 in {recording_dir.name}"

        extraction_root = Path(extraction_root)
        extraction_dir = extraction_root / recording_dir.name
        extraction_dir.mkdir(exist_ok=True)

        manual_csv_root = Path(manual_csv_root)
        manual_lib_csv_dir = manual_csv_root / recording_dir.name
        manual_lib_csv_dir.mkdir(exist_ok=True)

        model_csv_root = Path(model_csv_root)
        models_lib_csv_dir = model_csv_root / recording_dir.name
        models_lib_csv_dir.mkdir(exist_ok=True)


        # --- Section times ---
        rowset = target_sections[target_sections["recording id"] == recording_id]
        if rowset.empty:
            print(f"⚠️ No '{section_name}' section for {recording_id}")
            section_start, section_end = None, None
        else:
            row = rowset.iloc[0]
            section_start = int(row["section start time [ns]"])
            section_end = int(row["section end time [ns]"])

        # --- Build entry ---
        recordings_info[recording_id] = {
            "rec_name": recording_name,
            "start_time": start_time,
            "recording_root": str(recording_dir),
            "mp4_path": str(mp4_files[0]),
            "timestamps_path": str(timestamps_path),
            "extraction_dir": str(extraction_dir),
            "manual_csv_dir": str(manual_lib_csv_dir),
            "model_csv_dir": str(models_lib_csv_dir),
            "section_start_time_ns": section_start,
            "section_end_time_ns": section_end
        }

    # Save JSON output
    with open(json_path, "w") as f:
        json.dump(recordings_info, f, indent=2)

    print(f"✅ Saved recordings_info.json at {json_path}")

