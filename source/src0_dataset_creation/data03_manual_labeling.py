import os
import csv
import cv2
import json
from pathlib import Path
import pandas as pd

def manual_classify_frames_from_csv(recordings_info, rec_subset):
    """
    Manually classify frames based on deduplicated_frames.csv.
    Reconstructs frame filenames as <timestamp>_<suffix>.jpg and presents them for manual review.
    Saves results to <recording_dir>/manual_class.csv with columns: ['frame', 'is_face'].

    Parameters
    ----------
    recordings_info : dict
        Dictionary of recordings (from recordings_info.json).
    rec_subset : list[str], optional
        List of recording IDs to process. If None, all recordings are processed.
    """
    for rec_id, rec_dict in recordings_info.items():
        if rec_subset and rec_id not in rec_subset:
            continue

        extracted_frames = Path(rec_dict["extraction_dir"])
        manual_csv = Path(rec_dict["manual_csv_dir"])
        dedup_csv_path = manual_csv / "deduplicated_frames.csv"
        save_path = manual_csv / "manual_class.csv"

        if not dedup_csv_path.exists():
            print(f"⚠️ Skipping {rec_id} — no deduplicated_frames.csv found.")
            continue

        print(f"\n🧠 Starting manual classification for {rec_id} ({rec_dict.get('rec_name', '')})")

        # --- Load existing progress ---
        classifications = {}
        if save_path.exists():
            with open(save_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    classifications[row["frame"]] = int(row["is_face"])
            print(f"📄 Loaded {len(classifications)} existing classifications for {rec_id}.")

        # --- Load deduplicated CSV ---
        df = pd.read_csv(dedup_csv_path)
        if df.empty:
            print(f"⚠️ No frames in deduplicated_frames.csv for {rec_id}")
            continue

        # --- Construct filenames ---
        df["frame_name"] = df.apply(
            lambda r: f"{int(r['timestamp [ns]'])}_{int(r['suffix'])}.jpg",
            axis=1
        )

        images = df["frame_name"].tolist()
        print(f"🖼️ Found {len(images)} frames in deduplicated_frames.csv")

        if not images:
            continue

        cv2.namedWindow("Frame Review", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame Review", 800, 600)

        # --- Iterate over images ---
        for frame_name in images:
            if frame_name in classifications:
                continue  # skip already classified

            img_path = extracted_frames / frame_name
            if not img_path.exists():
                print(f"⚠️ Missing image file for {frame_name} — skipping")
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"⚠️ Could not read {frame_name}")
                continue

            cv2.imshow("Frame Review", frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("y"):
                classifications[frame_name] = 1
                print(f"✅ Face: {frame_name}")
            elif key == ord("n"):
                classifications[frame_name] = 0
                print(f"❌ Non-face: {frame_name}")
            elif key == ord("q"):
                print("💾 Exiting and saving progress...")
                break
            else:
                print("ℹ️ Press 'y' for face, 'n' for non-face, 'q' to quit.")
                continue

        # --- Save progress ---
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "is_face"])
            writer.writeheader()
            for frame, is_face in classifications.items():
                writer.writerow({"frame": frame, "is_face": is_face})

        cv2.destroyAllWindows()
        print(f"✅ Saved classifications for {rec_id} at {save_path}\n")
