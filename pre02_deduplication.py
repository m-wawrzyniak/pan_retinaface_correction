import pandas as pd
from pathlib import Path

from P01_config import EXTRACTION_DIR
from P02_model_parameters import TIME_THRESHOLD_NS

def base_timestamp(fname):
    # e.g., 1753444884755327156_2.jpg -> 1753444884755327156
    return int(fname.stem.split("_")[0])


def deduplicate_folder(folder_path: Path):
    print(f"\nProcessing folder: {folder_path.name}")

    # --- Step 1: List images ---
    all_files = sorted(folder_path.glob("*.jpg"), key=base_timestamp)

    if not all_files:
        print("No images found, skipping.")
        return

    # --- Step 2: Keep only timestamps far enough apart ---
    keep_timestamps = []
    last_ts = None

    for f in all_files:
        ts = base_timestamp(f)
        if last_ts is None or ts - last_ts >= TIME_THRESHOLD_NS:
            keep_timestamps.append(ts)
            last_ts = ts
        else:
            # Duplicate: delete the image
            f.unlink()

    print(f"Kept {len(keep_timestamps)} timestamps out of {len(all_files)} images")

    # --- Step 3: Filter CSV ---
    csv_path = folder_path / "face_frames.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df_filtered = df[df["timestamp [ns]"].isin(keep_timestamps)]
        df_filtered.to_csv(folder_path / "face_frames_filtered.csv", index=False)
        print(f"Filtered CSV rows: {len(df_filtered)} / {len(df)}")
    else:
        print("No face_frames.csv found, skipping CSV filtering.")


if __name__ == "__main__":
    extract_dir = Path(EXTRACTION_DIR)
    for subfolder in extract_dir.iterdir():
        if subfolder.is_dir():
            deduplicate_folder(subfolder)