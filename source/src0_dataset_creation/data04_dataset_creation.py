import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

from source.utilities import u01_dir_structure as u01

from config import P01_extraction_config as P01
from config import P02_model_config as P02

def split_dataset_with_constant_ratio(
        recordings_info,
        train_ratio,
        val_ratio,
        test_ratio,
        out_dir,
        random_state
    ):
    """
    Load all samples from valid recordings, then split train/val/test
    while preserving the SAME positive/negative ratio in all splits.
    """

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1."

    # --------------------------------------------------
    # 1) Load all valid CSVs
    # --------------------------------------------------
    all_rows = []

    for rec_id, rec_dict in recordings_info.items():

        # skip recordings without section_start_time_ns
        if rec_dict["section_start_time_ns"] is None:
            print(f"Skipping {rec_id}: no section_start_time_ns")
            continue

        manual_dir = Path(rec_dict["manual_csv_dir"])
        csv_path = manual_dir / "manual_class.csv"

        extraction_dir = Path(rec_dict["extraction_dir"])

        if not csv_path.exists():
            print(f"Missing {csv_path}, skipping {rec_id}")
            continue

        df = pd.read_csv(csv_path)
        df["path"] = df["frame"].apply(lambda f: str(extraction_dir / f))
        df["recording_id"] = rec_id
        all_rows.append(df)

    if not all_rows:
        raise ValueError("No valid recordings with manual_class.csv found.")

    df = pd.concat(all_rows, ignore_index=True)

    # --------------------------------------------------
    # 2) Split into positive / negative pools
    # --------------------------------------------------
    df_pos = df[df["is_face"] == 1]
    df_neg = df[df["is_face"] == 0]

    print(f"Loaded {len(df)} items:")
    print(f" Positives: {len(df_pos)}")
    print(f" Negatives: {len(df_neg)}")

    # --------------------------------------------------
    # 3) Split positives and negatives separately
    # --------------------------------------------------
    # positives
    pos_train, pos_tmp = train_test_split(
        df_pos, test_size=(1 - train_ratio),
        random_state=random_state, shuffle=True
    )
    pos_val, pos_test = train_test_split(
        pos_tmp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state, shuffle=True
    )

    # negatives
    neg_train, neg_tmp = train_test_split(
        df_neg, test_size=(1 - train_ratio),
        random_state=random_state, shuffle=True
    )
    neg_val, neg_test = train_test_split(
        neg_tmp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state, shuffle=True
    )

    # --------------------------------------------------
    # 4) Merge splits and shuffle
    # --------------------------------------------------
    train_df = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df   = pd.concat([pos_val,   neg_val]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df  = pd.concat([pos_test,  neg_test]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # --------------------------------------------------
    # 5) Print final distributions
    # --------------------------------------------------
    def dist(name, df):
        p = (df["is_face"] == 1).sum()
        n = (df["is_face"] == 0).sum()
        print(f"{name}: {len(df)} items ({p} pos, {n} neg, pos_ratio={p/len(df):.3f})")

    dist("Train", train_df)
    dist("Val", val_df)
    dist("Test", test_df)

    out_dir = Path(out_dir)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    return train_df, val_df, test_df

if __name__ == "__main__":
    paths_dict = u01.build_absolute_paths(
        root=P01.ROOT,
        classifier_name=P01.CLASSIFIER_NAME,
        dataset_name=P01.DATASET_NAME,
        project_json_path=P01.PROJECT_STRUCT
    )

    with open(paths_dict['data']['datasets'][P01.DATASET_NAME]['recordings_info.json'], "r") as f:
        recordings_info = json.load(f)

    train_df, val_df, test_df = split_dataset_with_constant_ratio(
        recordings_info=recordings_info,
        train_ratio=P02.TRAIN_RATIO,
        val_ratio=P02.VAL_RATIO,
        test_ratio=P02.TEST_RATIO,
        out_dir=paths_dict['data']['classifiers'][P01.CLASSIFIER_NAME]['sets']['_dir'],
        random_state=P02.SEED
    )
