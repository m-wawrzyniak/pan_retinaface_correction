import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

from P01_config import LABELS_CSV, LABELED_DIR, DATASET_DIR

from P02_model_parameters import SPLIT_RATIOS

# TODO: Should the dataset be balanced? What else should be done here?

# Paths
labels_csv = Path(LABELS_CSV)   # your labels file
images_dir = Path(LABELED_DIR) # root with recording_id folders
dataset_dir = Path(DATASET_DIR)  # output dataset root

# --- Step 1: Read CSV ---
df = pd.read_csv(labels_csv)

# The source file lives in: extracted_faces/recording_id/timestamp.jpg
df["src_path"] = df.apply(lambda row: images_dir / str(row.recording_id) / f"{row.timestamp}.jpg", axis=1)

# To make filenames unique after copying:
df["new_name"] = df.apply(lambda row: f"{row.recording_id}_{row.timestamp}.jpg", axis=1)

# --- Step 2: Train/Val/Test Split ---
train_df, temp_df = train_test_split(df, test_size=(1-SPLIT_RATIOS[0]),
                                     stratify=df["is_face"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=SPLIT_RATIOS[2]/(SPLIT_RATIOS[1]+SPLIT_RATIOS[2]),
                                   stratify=temp_df["is_face"], random_state=42)

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

# --- Step 3: Create folders ---
for split in splits:
    for cls in ["face", "nonface"]:
        (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

# --- Step 4: Copy or Symlink files ---
def link_files(subset, split_name, use_symlinks=False):
    for _, row in subset.iterrows():
        if row["is_face"] == 1:
            dst = dataset_dir / split_name / "face" / row["new_name"]
        else:
            dst = dataset_dir / split_name / "nonface" / row["new_name"]

        try:
            if use_symlinks:
                dst.symlink_to(row["src_path"].resolve())
            else:
                shutil.copy(row["src_path"], dst)
        except FileExistsError:
            pass
        except FileNotFoundError:
            print(f"⚠️ Missing file: {row['src_path']}")

for split_name, subset in splits.items():
    link_files(subset, split_name, use_symlinks=False)  # set True to save space

print("✅ Dataset prepared at:", dataset_dir.resolve())
