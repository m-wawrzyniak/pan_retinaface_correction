import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path

from EyetrackerCNN import EyetrackerCNN

from config.P01_config import EXTRACTION_DIR, MODEL_PATH, CLASSIFICATION_CSV


classification_csv = Path(CLASSIFICATION_CSV)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Transform / preprocessing (match training) ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# --- Load model ---
model = EyetrackerCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Iterate over subfolders ---
results = []

for rec_folder in Path(EXTRACTION_DIR).iterdir():
    if rec_folder.is_dir():
        rec_id = rec_folder.name
        img_paths = list(rec_folder.glob("*.jpg"))
        total_frames = len(img_paths)
        print(f"➡️ Processing recording: {rec_id} ({total_frames} frames)")

        for idx, img_path in enumerate(img_paths, 1):
            timestamp = img_path.stem
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = model(img_tensor).item()
                    is_face = int(prob > 0.5)

                results.append({
                    "recording_id": rec_id,
                    "timestamp": timestamp,
                    "is_face": is_face,
                    "prob": prob
                })

                # Log progress per recording as percentage
                if idx % max(1, total_frames // 10) == 0 or idx == total_frames:
                    percent = (idx / total_frames) * 100
                    print(f"    {percent:.1f}% of frames processed in {rec_id}")

            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")

        print(f"✅ Finished recording: {rec_id}")

# --- Save to CSV ---
df = pd.DataFrame(results)
df.to_csv(classification_csv, index=False)
print(f"✅ Classification saved to {classification_csv}")
