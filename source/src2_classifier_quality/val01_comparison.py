import pandas as pd
from pathlib import Path
import json
from pprint import pprint

from source.utilities import u01_dir_structure as u01

from config import P01_extraction_config as P01
from config import P02_model_config as P02

def build_full_labels_csv(recordings_info: dict, out_csv: str = None):
    """
    Builds a full labels dataframe containing:
      - frame_path
      - frame (filename)
      - recording_id
      - ground_truth  (from manual_class.csv)
      - primary_pred  (always 1)
      - secondary_pred (from model_class.csv)

    Parameters
    ----------
    recordings_info : dict
        The dictionary with paths (like the one you already generate)
    out_csv : str or None
        If provided, saves the final CSV to this path.

    Returns
    -------
    pd.DataFrame
    """

    all_rows = []

    for rec_id, rec_dict in recordings_info.items():

        # --- 1. Manual CSV ---
        manual_dir = Path(rec_dict["manual_csv_dir"])
        manual_csv_path = manual_dir / "manual_class.csv"

        if not manual_csv_path.exists():
            print(f"[skip] Missing manual_class.csv for {rec_id}")
            continue

        # Load manual ground truth
        df_manual = pd.read_csv(manual_csv_path)

        if "frame" not in df_manual.columns or "is_face" not in df_manual.columns:
            print(f"[skip] Invalid manual CSV format in {manual_csv_path}")
            continue

        # Build full frame path (using extraction_dir)
        extraction_dir = Path(rec_dict["extraction_dir"])
        df_manual["frame_path"] = df_manual["frame"].apply(lambda f: str(extraction_dir / f))
        df_manual["ground_truth"] = df_manual["is_face"].astype(int)
        df_manual["recording_id"] = rec_id
        df_manual = df_manual[["frame", "frame_path", "recording_id", "ground_truth"]]

        # --- 2. Secondary model CSV ---
        model_dir = Path(rec_dict["model_csv_dir"])
        model_csv_path = model_dir / "model_class.csv"

        if not model_csv_path.exists():
            print(f"[skip] Missing model_class.csv for {rec_id}")
            continue

        df_model = pd.read_csv(model_csv_path)

        if "frame" not in df_model.columns or "is_face" not in df_model.columns:
            print(f"[skip] Invalid model_class.csv format in {model_csv_path}")
            continue

        df_model.rename(columns={"is_face": "secondary_pred"}, inplace=True)
        df_model["secondary_pred"] = df_model["secondary_pred"].astype(int)

        # --- 3. Merge on frame name ---
        merged = df_manual.merge(df_model[["frame", "secondary_pred"]],
                                 on="frame", how="left")

        # If model did not classify a frame → warn (shouldn't happen)
        missing_sec = merged["secondary_pred"].isna().sum()
        if missing_sec > 0:
            print(f"[warn] {missing_sec} frames in {rec_id} missing secondary predictions")
            merged["secondary_pred"] = merged["secondary_pred"].fillna(0)

        # --- 4. Add primary_pred = 1 ---
        merged["primary_pred"] = 1

        all_rows.append(merged)

    # --- Concatenate everything ---
    if not all_rows:
        print("No rows found. Returning empty DataFrame.")
        return pd.DataFrame()

    full_df = pd.concat(all_rows, ignore_index=True)

    # Optional: save
    if out_csv:
        full_df.to_csv(out_csv, index=False)
        print(f"[OK] Saved full labels CSV → {out_csv}")

    return full_df


def compute_improvement_metrics(df):
    """
    Computes FPRR, FPRs, precision boost, and false negatives/recall comparing the primary and secondary model.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        - ground_truth (0/1)
        - primary_pred (0/1)
        - secondary_pred (0/1)

    Returns
    -------
    dict
        {
            "primary_fpr": float,
            "secondary_fpr": float,
            "fpr_reduction_ratio": float,
            "primary_precision": float,
            "secondary_precision": float,
            "precision_boost": float,
            "false_negatives": int,
            "total_faces": int,
            "secondary_recall": float,
            "recall_loss": float
        }
    """

    # ----- Utility functions -----
    def precision(tp, fp):
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def fpr(fp, tn):
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # ----- PRIMARY model confusion counts -----
    p_tp = ((df.primary_pred == 1) & (df.ground_truth == 1)).sum()
    p_fp = ((df.primary_pred == 1) & (df.ground_truth == 0)).sum()
    p_tn = ((df.primary_pred == 0) & (df.ground_truth == 0)).sum()

    # ----- SECONDARY model confusion counts -----
    s_tp = ((df.secondary_pred == 1) & (df.ground_truth == 1)).sum()
    s_fp = ((df.secondary_pred == 1) & (df.ground_truth == 0)).sum()
    s_tn = ((df.secondary_pred == 0) & (df.ground_truth == 0)).sum()
    s_fn = ((df.secondary_pred == 0) & (df.ground_truth == 1)).sum()

    # ----- Metrics -----
    primary_fpr = fpr(p_fp, p_tn)
    secondary_fpr = fpr(s_fp, s_tn)
    fpr_reduction_ratio = ((primary_fpr - secondary_fpr) / primary_fpr) if primary_fpr > 0 else 0.0

    primary_precision = precision(p_tp, p_fp)
    secondary_precision = precision(s_tp, s_fp)
    precision_boost = secondary_precision - primary_precision

    # ----- False negatives / recall -----
    total_faces = s_tp + s_fn
    secondary_recall = s_tp / total_faces if total_faces > 0 else 0.0
    recall_loss = 1 - secondary_recall  # compared to primary recall = 1.0

    return {
        "primary_fpr": primary_fpr,
        "secondary_fpr": secondary_fpr,
        "fpr_reduction_ratio": fpr_reduction_ratio,
        "primary_precision": primary_precision,
        "secondary_precision": secondary_precision,
        "precision_boost": precision_boost,
        "secondary_recall": secondary_recall,
        "recall_loss": recall_loss
    }


if __name__ == '__main__':
    paths_dict = u01.build_absolute_paths(
        root=P01.ROOT,
        classifier_name=P01.CLASSIFIER_NAME,
        dataset_name=P01.DATASET_NAME,
        project_json_path=P01.PROJECT_STRUCT
    )

    with open(paths_dict['data']['datasets'][P01.DATASET_NAME]['recordings_info.json'], "r") as f:
        recordings_info = json.load(f)

    comparison_csv_path = paths_dict['data']['classifiers'][P01.CLASSIFIER_NAME]['validation_metrics']['comparison_class.csv']
    build_full_labels_csv(recordings_info,
                          out_csv=comparison_csv_path)

    comp_df = pd.read_csv(comparison_csv_path)
    improvements_results = compute_improvement_metrics(comp_df)
    pprint(improvements_results)
