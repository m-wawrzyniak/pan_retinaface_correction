from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import torch
from torchvision import transforms
from PIL import Image

from source.utilities import u00_html_visuals as v00
from source.src1_classifier.Classifier import FaceVerifierCNN

import config.P01_extraction_config as P01
import config.P02_model_config as P02

def load_model(model_path, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaceVerifierCNN(input_size=input_size).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    # Your training script saves "model_state_dict"
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"model_state_dict missing in checkpoint! Keys: {checkpoint.keys()}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def classify_frame(model, device, img_path, transform, dec_threshold):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        prob = torch.sigmoid(logits).item()

    return 1 if prob >= dec_threshold else 0


def run_inference_on_recordings(model_path, rec_subset, recordings_info_json, input_size, dec_threshold):

    # Load model
    model, device = load_model(model_path, input_size)

    # Build the SAME eval transform as in training
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load metadata
    with open(recordings_info_json, "r") as f:
        recordings_info = json.load(f)

    for rec_id in rec_subset:

        if rec_id not in recordings_info:
            print(f"⚠ Missing in recordings_info.json: {rec_id}")
            continue

        extraction_dir = Path(recordings_info[rec_id]["extraction_dir"])

        if not extraction_dir.exists():
            print(f"⚠ extraction_dir does not exist: {extraction_dir}")
            continue

        print(f"\n📂 Processing: {rec_id}")

        jpg_files = sorted(
            p for p in extraction_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )

        if not jpg_files:
            print("  ⚠ No image files — skipping")
            continue

        results = []

        for frame_path in tqdm(jpg_files, desc=f"{rec_id}", unit="frame"):
            pred = classify_frame(model, device, frame_path, transform, dec_threshold=dec_threshold)
            results.append((frame_path.name, pred))

        out_csv = extraction_dir / "model_class.csv"
        df = pd.DataFrame(results, columns=["frame", "is_face"])
        df.to_csv(out_csv, index=False)

        print(f"  ✔ Saved predictions → {out_csv}")



if __name__ == "__main__":

    run_inference_on_recordings(
        model_path="/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/data/classifiers/class_v00/best_model.pth", # <-- update
        input_size=P02.CNN_INPUT_SIZE,
        dec_threshold=P02.PROB_THRESHOLD,
        rec_subset=P01.REC_SUBSET,
        recordings_info_json = P01.RECORDINGS_INFO_PATH
    )
    with open(P01.RECORDINGS_INFO_PATH, "r") as f:
        recordings_info = json.load(f)
    for rec, rec_dict in recordings_info.items():
        if rec_dict["section_start_time_ns"] != None:
            v00.export_html_paginated(
                name='model',
                csv_path= Path(rec_dict['extraction_dir']) / 'model_class.csv'
            )