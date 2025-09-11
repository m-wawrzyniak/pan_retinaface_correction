import os
from pprint import pprint

from P01_config import REC_DIR
from P01_config import EXTRACTION_DIR

def fetch_rec_paths():
    rec_paths = {}
    for folder_name in os.listdir(REC_DIR):
        folder_path = os.path.join(REC_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # sections.csv
        sections_path = os.path.join(folder_path, "sections.csv")
        if not os.path.isfile(sections_path):
            continue  # skip if invalid recording folder

        # find the only subfolder
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        if len(subfolders) != 1:
            raise RuntimeError(f"Expected exactly one subfolder in {folder_path}, found {subfolders}")
        subfolder_path = os.path.join(folder_path, subfolders[0])

        # world_timestamps.csv
        timestamps_path = os.path.join(subfolder_path, "world_timestamps.csv")
        if not os.path.isfile(timestamps_path):
            raise FileNotFoundError(f"world_timestamps.csv not found in {subfolder_path}")

        # mp4 file
        mp4_files = [f for f in os.listdir(subfolder_path) if f.endswith(".mp4")]
        if len(mp4_files) != 1:
            raise RuntimeError(f"Expected exactly one .mp4 in {subfolder_path}, found {mp4_files}")
        video_path = os.path.join(subfolder_path, mp4_files[0])

        # extraction folder
        extract_path = os.path.join(EXTRACTION_DIR, folder_name)

        # store in dict
        rec_paths[folder_name] = {
            "sections": sections_path,
            "timestamps": timestamps_path,
            "video": video_path,
            "extract": extract_path
        }

    return rec_paths

if __name__ == "__main__":
    pprint(fetch_rec_paths())
