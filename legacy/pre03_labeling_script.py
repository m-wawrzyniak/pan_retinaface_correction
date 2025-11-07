import cv2
import csv
from pathlib import Path


from config.P01_config import LABELED_DIR, LABELS_CSV

# --- CONFIG ---
labeled_dir = Path(LABELED_DIR)
label_csv = Path(LABELS_CSV)

# --- OUTPUT CSV SETUP ---
if not label_csv.exists():
    with open(label_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording_id", "timestamp", "is_face"])

# --- MAIN LOOP ---
with open(label_csv, "a", newline="") as f:
    writer = csv.writer(f)

    # iterate through subfolders (recording_ids)
    for rec_dir in sorted(labeled_dir.iterdir()):
        if not rec_dir.is_dir():
            continue

        recording_id = rec_dir.name
        print(f"Processing recording: {recording_id}")

        # iterate through images
        for img_path in sorted(rec_dir.glob("*.jpg")):
            timestamp = img_path.stem  # filename without extension

            # show image
            img = cv2.imread(str(img_path))
            cv2.imshow("Labeling", img)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("t") or key == ord("T"):
                is_face = 1
            elif key == ord("f") or key == ord("F"):
                is_face = 0
            elif key == 27:  # ESC to quit
                print("Exiting early.")
                cv2.destroyAllWindows()
                exit(0)
            else:
                print("Invalid key, skipping image.")
                continue

            # write row
            writer.writerow([recording_id, timestamp, is_face])
            f.flush()  # ensure progress is saved

cv2.destroyAllWindows()
