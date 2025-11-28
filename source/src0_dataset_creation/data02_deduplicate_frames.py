import pandas as pd
from pathlib import Path
import json

from config import P01_extraction_config as P01

def deduplicate_face_frames_csv(recordings_info, rec_subset, threshold):
    """
    Deduplicate frames in face_frames.csv based on temporal closeness.
    Applies deduplication independently per suffix.
    Saves a new CSV deduplicated_frames.csv in the same folder.

    Parameters
    ----------
    recordings_info : dict
        Dictionary containing info for all recordings including 'extraction_dir'.
    rec_subset : list[str], optional
        List of recording_ids to process. If None, all recordings are processed.
    threshold : float
        Minimum time difference in seconds between frames to keep them.
    """
    for rec_id, rec_dict in recordings_info.items():
        if rec_subset and rec_id not in rec_subset:
            continue

        extraction_dir = rec_dict["extraction_dir"]
        face_csv_path = Path(extraction_dir) / "face_frames.csv"
        manual_dir = rec_dict["manual_csv_dir"]
        dedup_csv_path = Path(manual_dir) / "deduplicated_frames.csv"

        df = pd.read_csv(face_csv_path)
        if df.empty:
            print(f"⚠️ No face frames for {rec_id}")
            continue

        df = df.sort_values("timestamp [ns]").reset_index(drop=True)
        dedup_rows = []

        # --- deduplicate independently per suffix ---
        for suffix, group in df.groupby("suffix"):
            group = group.reset_index(drop=True)
            timestamps = group["timestamp [ns]"].values

            keep_idxs = [0]  # always keep first
            last_kept_idx = 0

            for i in range(1, len(group) - 1):
                t_prev = timestamps[last_kept_idx]
                t_next = timestamps[i + 1]
                delta = (t_next - t_prev) / 1e9
                if delta >= threshold:
                    keep_idxs.append(i)
                    last_kept_idx = i

            keep_idxs.append(len(group) - 1)  # always keep last
            keep_idxs = sorted(set(keep_idxs))

            dedup_rows.append(group.iloc[keep_idxs])

        dedup_df = pd.concat(dedup_rows, ignore_index=True)
        dedup_df.to_csv(dedup_csv_path, index=False)
        print(f"✅ Deduplicated {len(df)} → {len(dedup_df)} frames for {rec_id}")


if __name__ == "__main__":
    with open(P01.RECORDINGS_INFO_PATH, "r") as f:
        recordings_info = json.load(f)
    deduplicate_face_frames_csv(
        recordings_info=recordings_info,
        rec_subset=P01.REC_SUBSET,
        threshold=P01.DEDUP_THRESHOLD
    )
