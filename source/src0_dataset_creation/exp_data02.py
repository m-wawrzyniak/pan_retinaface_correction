import pandas as pd
import numpy as np
from pathlib import Path
import json
from PIL import Image

from sklearn.cluster import KMeans
from facenet_pytorch import InceptionResnetV1
import torch

from config import P01_extraction_config as P01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def compute_embedding(img_path):
    """Load image → resize → tensor → embedding vector."""
    try:
        img = Image.open(img_path).convert("RGB").resize((160, 160))
    except:
        return None

    t = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    t = (t - 0.5) / 0.5   # normalize roughly to facenet expectations
    t = t.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = embedding_model(t).cpu().numpy()[0]
    return emb


def importance_sample(df, embeddings, n_clusters, min_per_cluster, max_total, seed):
    """
    Clusters embeddings and picks samples:
      - At least min_per_cluster per cluster
      - More from larger clusters proportionally
    """
    if len(df) <= max_total:
        return df.copy()

    # clustering
    km = KMeans(n_clusters=n_clusters, n_init='auto')
    clusters = km.fit_predict(embeddings)

    df = df.copy()
    df["cluster"] = clusters

    sampled_indices = []

    for c in range(n_clusters):
        cluster_df = df[df.cluster == c]
        size = len(cluster_df)

        if size == 0:
            continue

        base = min(min_per_cluster, size)

        # proportional scaling
        extra = int((size / len(df)) * (max_total - n_clusters * min_per_cluster))
        k = min(size, base + extra)

        sampled_indices.extend(cluster_df.sample(k, random_state=seed).index.tolist())

    sampled = df.loc[sorted(set(sampled_indices))]

    # safety: cap total
    if len(sampled) > max_total:
        sampled = sampled.sample(max_total, random_state=seed)

    return sampled


# --------------------------------------------------------
# 3. Original: Deduplication + Augmented pipeline
# --------------------------------------------------------
def sample_face_frames_csv(recordings_info,
                           rec_subset,
                           threshold,
                           n_clusters,
                           min_per_cluster,
                           max_total,
                           seed
                           ):
    """
    Deduplicate, compute embeddings, cluster, sample.
    Writes:
      - deduplicated_frames.csv
      - sampled_frames.csv
      - embeddings.npy
    """
    for rec_id, rec_dict in recordings_info.items():
        if rec_subset and rec_id not in rec_subset:
            continue

        extraction_dir = rec_dict["extraction_dir"]
        face_csv_path = Path(extraction_dir) / "face_frames.csv"
        manual_dir = Path(rec_dict["manual_csv_dir"])
        dedup_csv_path = manual_dir / "deduplicated_frames.csv"
        emb_path = manual_dir / "embeddings.npy"
        sample_csv_path = manual_dir / "sampled_frames.csv"

        df = pd.read_csv(face_csv_path)
        if df.empty:
            print(f"⚠️ No face frames for {rec_id}")
            continue

        df = df.sort_values("timestamp [ns]").reset_index(drop=True)
        dedup_rows = []

        # --- deduplicate by suffix ---
        for suffix, group in df.groupby("suffix"):
            group = group.reset_index(drop=True)
            timestamps = group["timestamp [ns]"].values

            keep_idxs = [0]
            last_kept_idx = 0

            for i in range(1, len(group) - 1):
                t_prev = timestamps[last_kept_idx]
                t_next = timestamps[i + 1]
                delta = (t_next - t_prev) / 1e9
                if delta >= threshold:
                    keep_idxs.append(i)
                    last_kept_idx = i

            keep_idxs.append(len(group) - 1)
            keep_idxs = sorted(set(keep_idxs))

            dedup_rows.append(group.iloc[keep_idxs])

        dedup_df = pd.concat(dedup_rows, ignore_index=True)
        dedup_df.to_csv(dedup_csv_path, index=False)
        print(f"✅ Deduplicated {len(df)} → {len(dedup_df)} frames for {rec_id}")

        # --------------------------------------------------------
        # Compute embeddings for dedup frames
        # --------------------------------------------------------
        embeddings = []
        valid_indices = []

        for idx, row in dedup_df.iterrows():
            emb = compute_embedding(row["frame_path"])
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(idx)

        if len(embeddings) == 0:
            print(f"⚠️ No valid embeddings for {rec_id}")
            continue

        embeddings = np.vstack(embeddings)
        np.save(emb_path, embeddings)
        print(f"🔷 Saved embeddings for {rec_id}: {emb_path}")

        valid_df = dedup_df.iloc[valid_indices].reset_index(drop=True)

        # --------------------------------------------------------
        # importance sampling
        # --------------------------------------------------------
        sampled_df = importance_sample(valid_df, embeddings,
                                       n_clusters=n_clusters,
                                       min_per_cluster=min_per_cluster,
                                       max_total=max_total,
                                       seed=seed)

        sampled_df.to_csv(sample_csv_path, index=False)
        print(f"🎯 Sampled {len(sampled_df)} frames → {sample_csv_path}")

