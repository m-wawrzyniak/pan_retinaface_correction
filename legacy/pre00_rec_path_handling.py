import os
import json
from pathlib import Path
from pprint import pprint

from config.P01_config import REC_DIR
from config.P01_config import EXTRACTION_DIR

def fetch_rec_paths():
    rec_paths = {}
    for date_folder in os.listdir(REC_DIR):
        date_path = Path(REC_DIR) / date_folder
        if not date_path.is_dir():
            continue

        # info.json
        info_path = date_path / "info.json"
        if not info_path.is_file():
            continue

        with open(info_path, "r") as f:
            info = json.load(f)
        recording_id = info.get("recording_id")
        if recording_id is None:
            raise KeyError(f"recording_id missing in {info_path}")

        # video file
        video_path = date_path / "Neon Scene Camera v1 ps1.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found in {date_path}")

        # extraction path based on recording_id
        extract_path = Path(EXTRACTION_DIR) / recording_id

        # store in dict
        rec_paths[date_folder] = {
            "recording_id": recording_id,
            "video": str(video_path),
            "extract": str(extract_path)
        }

    return rec_paths

if __name__ == "__main__":
    pprint(fetch_rec_paths())
