import os
import json
from pathlib import Path

def collect_recording_ids(root_dir: str):
    """
    Scans all subdirectories of `root_dir`, looks for an info.json file,
    extracts 'recording_id', and returns a list of all found IDs.
    """
    root = Path(root_dir)
    recording_ids = []

    for sub in root.iterdir():
        if not sub.is_dir():
            continue

        info_path = sub / "info.json"
        if not info_path.exists():
            continue

        try:
            with open(info_path, "r") as f:
                data = json.load(f)
            rec_id = data.get("recording_id")
            if rec_id:
                recording_ids.append(rec_id)
        except Exception as e:
            print(f"⚠️ Could not read {info_path}: {e}")

    return recording_ids

root = '/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Timeseries Data + Scene Video/'
print(collect_recording_ids(root))