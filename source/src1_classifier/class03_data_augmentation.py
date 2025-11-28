import os
import json
import pandas as pd
from pathlib import Path
from source.utilities import u01_dir_structure as u01

import config.P01_extraction_config as P01

def prune_face_frames(rec_subset, recordings_info_path):
    with open(recordings_info_path, "r") as f:
        recordings_info = json.load(f)

    for rec_id in rec_subset:
        if rec_id not in recordings_info:
            print(f"⚠️ {rec_id} not in recordings_info.json → skipping")
            continue

        rec_dict = recordings_info[rec_id]
        face_csv_path = Path(rec_dict["extraction_dir"]) / "face_frames.csv"
        model_csv_path = Path(rec_dict["model_csv_dir"]) / "model_class.csv"
        augmented_csv_path = Path(rec_dict["model_csv_dir"]) / "augmented_face_frames.csv"

        if not face_csv_path.exists():
            print(f"⚠️ face_frames.csv missing for {rec_id}")
            continue
        if not model_csv_path.exists():
            print(f"⚠️ model_class.csv missing for {rec_id}")
            continue

        face_df = pd.read_csv(face_csv_path)
        model_df = pd.read_csv(model_csv_path)

        # Construct actual frame names from timestamp + suffix
        face_df["frame_name"] = face_df["timestamp [ns]"].astype(str) + "_" + face_df["suffix"].astype(str) + ".jpg"

        # Keep only frames classified as is_face=1
        face_frames_set = set(model_df[model_df["is_face"] == 1]["frame"])
        filtered_df = face_df[face_df["frame_name"].isin(face_frames_set)].copy()

        # Drop helper column and save
        filtered_df.drop(columns=["frame_name"], inplace=True)
        filtered_df.to_csv(augmented_csv_path, index=False)
        print(f"✅ {rec_id} → augmented_face_frames.csv ({len(filtered_df)} frames)")

def aggregate_augmented_face_detections(rec_subset, recordings_info_path, output_dir):
    with open(recordings_info_path, "r") as f:
        recordings_info = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "augmented_face_detections.csv"

    all_frames = []

    for rec_id in rec_subset:
        if rec_id not in recordings_info:
            print(f"⚠️ {rec_id} not in recordings_info.json → skipping")
            continue

        rec_dict = recordings_info[rec_id]
        augmented_csv_path = Path(rec_dict["model_csv_dir"]) / "augmented_face_frames.csv"

        if not augmented_csv_path.exists():
            print(f"⚠️ {augmented_csv_path} missing → skipping {rec_id}")
            continue

        df = pd.read_csv(augmented_csv_path)
        df = df.drop(columns=["suffix"], errors="ignore")  # drop suffix if exists
        all_frames.append(df)

        print(f"✅ {rec_id}: {len(df)} frames added")

    if not all_frames:
        print("⚠️ No frames collected. Exiting.")
        return

    aggregated_df = pd.concat(all_frames, ignore_index=True)
    aggregated_df.to_csv(output_csv, index=False)
    print(f"✅ Aggregated CSV saved at {output_csv} ({len(aggregated_df)} frames)")

if __name__ == "__main__":
    paths_dict = u01.build_absolute_paths(
        root=P01.ROOT,
        classifier_name=P01.CLASSIFIER_NAME,
        dataset_name=P01.DATASET_NAME,
        project_json_path=P01.PROJECT_STRUCT
    )

    prune_face_frames(
        rec_subset=P01.REC_SUBSET,
        recordings_info_path=paths_dict['data']['datasets'][P01.DATASET_NAME]['recordings_info.json'])
    aggregate_augmented_face_detections(
        rec_subset=P01.REC_SUBSET,
        recordings_info_path=paths_dict['data']['datasets'][P01.DATASET_NAME]['recordings_info.json'],
        output_dir=P01.TIMESERIES_DATA)