import pandas as pd
from pathlib import Path
import json

from sklearn.model_selection import train_test_split

from config.P01_extraction_config import RECORDINGS_INFO_PATH
from config import P02_model_config as P02

def split_dataset_with_ratios(
        recordings_info,
        train_ratio,
        val_ratio,
        test_ratio,
        pos_fraction_train,
        pos_fraction_val,
        pos_fraction_test,
        csv_name="manual_class.csv",
        dataset_name='dummy'
    ):
    """
    Create train/val/test splits with controlled class ratios, but
    ONLY using recordings where section_start_time_ns is not None.

    Parameters
    ----------
    recordings_info : dict
        Your recordings metadata dict. Must contain:
        rec_dict["extraction_dir"]
        rec_dict["section_start_time_ns"]

    csv_name : str
        The manual label CSV file name in each extraction dir.

    pos_fraction_* : float
        Desired fraction of positive samples in each split.
    """

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1."

    # ---------------------------------------------
    # 1) Load and filter only valid recordings
    # ---------------------------------------------
    all_rows = []

    for rec_id, rec_dict in recordings_info.items():

        # skip bad recordings
        if rec_dict["section_start_time_ns"] is None:
            print(f"Skipping {rec_id}: no section_start_time_ns")
            continue

        extraction_dir = Path(rec_dict["extraction_dir"])
        csv_path = extraction_dir / csv_name

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

    # ---------------------------------------------
    # 2) Split into pos / neg pools
    # ---------------------------------------------
    df_pos = df[df["is_face"] == 1].copy()
    df_neg = df[df["is_face"] == 0].copy()

    print(f"Loaded {len(df)} items:")
    print(f"  Positives: {len(df_pos)}")
    print(f"  Negatives: {len(df_neg)}")

    # ---------------------------------------------
    # Helper for ratio-controlled sampling
    # ---------------------------------------------
    def create_split(df_pos, df_neg, total_size, pos_fraction):
        n_pos = int(total_size * pos_fraction)
        n_neg = total_size - n_pos

        take_pos = min(len(df_pos), n_pos)
        take_neg = min(len(df_neg), n_neg)

        sampled_pos = df_pos.sample(n=take_pos, replace=False, random_state=42)
        sampled_neg = df_neg.sample(n=take_neg, replace=False, random_state=42)

        # remove sampled from pools
        df_pos = df_pos.drop(sampled_pos.index)
        df_neg = df_neg.drop(sampled_neg.index)

        combined = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=42)
        return combined, df_pos, df_neg

    # ---------------------------------------------
    # 3) Determine split sizes
    # ---------------------------------------------
    N = len(df)
    N_train = int(N * train_ratio)
    N_val = int(N * val_ratio)
    N_test = N - N_train - N_val

    # ---------------------------------------------
    # 4) Create the splits
    # ---------------------------------------------
    train_df, df_pos, df_neg = create_split(df_pos, df_neg, N_train, pos_fraction_train)
    val_df,   df_pos, df_neg = create_split(df_pos, df_neg, N_val,   pos_fraction_val)
    test_df,  df_pos, df_neg = create_split(df_pos, df_neg, N_test,  pos_fraction_test)

    # ---------------------------------------------
    # 5) Save splits
    # ---------------------------------------------
    out_dir = Path(f"/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/data/datasets/{dataset_name}")
    out_dir.mkdir(exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("\nSaved splits:")
    print(out_dir / "train.csv")
    print(out_dir / "val.csv")
    print(out_dir / "test.csv")

    return train_df, val_df, test_df

def split_dataset_with_constant_ratio(
        recordings_info,
        train_ratio,
        val_ratio,
        test_ratio,
        random_state,
        csv_name="manual_class.csv",
        dataset_name="dummy"
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

        extraction_dir = Path(rec_dict["extraction_dir"])
        csv_path = extraction_dir / csv_name

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

    # --------------------------------------------------
    # 6) Save to disk
    # --------------------------------------------------
    out_dir = Path(f"/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/data/datasets/{dataset_name}")
    out_dir.mkdir(exist_ok=True, parents=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("\nSaved splits:")
    print(out_dir / "train.csv")
    print(out_dir / "val.csv")
    print(out_dir / "test.csv")

    return train_df, val_df, test_df

if __name__ == "__main__":
    with open(RECORDINGS_INFO_PATH, "r") as f:
        recordings_info = json.load(f)

    train_df, val_df, test_df = split_dataset_with_constant_ratio(
        recordings_info=recordings_info,
        csv_name="manual_class.csv",
        train_ratio=P02.TRAIN_RATIO,
        val_ratio=P02.VAL_RATIO,
        test_ratio=P02.TEST_RATIO,
        random_state=P02.SEED
    )

    '''
    # OLD
    train_df, val_df, test_df = split_dataset_with_ratios(
        recordings_info=recordings_info,
        csv_name="manual_class.csv",
        train_ratio=P02.TRAIN_RATIO,
        val_ratio=P02.VAL_RATIO,
        test_ratio=P02.TEST_RATIO,
        pos_fraction_train=P02.POS_FRAC_TRAIN,
        pos_fraction_val=P02.POS_FRAC_VAL,
        pos_fraction_test=P02.POS_FRAC_TEST
    )
    '''