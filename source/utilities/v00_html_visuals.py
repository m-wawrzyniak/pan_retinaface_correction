import pandas as pd
from pathlib import Path

from config.P01_config import EXTRACTION_DIR, CLASSIFICATION_CSV, HTML_PATH

# --- Paths ---
extraction_path = Path(EXTRACTION_DIR)
classification_csv_path = Path(CLASSIFICATION_CSV)
output_dir = Path(HTML_PATH)
output_dir.mkdir(exist_ok=True)

# --- Read classification ---
df = pd.read_csv(classification_csv_path)

# --- Separate faces / non-faces ---
faces_df = df[df["is_face"] == 0]
nonfaces_df = df[df["is_face"] == 1]

def create_html(df_subset, filename):
    html_file = output_dir / filename
    with open(html_file, "w") as f:
        f.write("<html><body>\n")
        f.write("<div style='display:flex; flex-wrap: wrap;'>\n")
        for _, row in df_subset.iterrows():
            rec_folder = extraction_path / row["recording_id"]
            img_path = rec_folder / f"{row['timestamp']}.jpg"
            if img_path.exists():
                # Display image with fixed size
                f.write(f"<div style='margin:2px;'>"
                        f"<img src='{img_path}' style='width:150px; height:150px;'/>"
                        f"</div>\n")
        f.write("</div></body></html>\n")
    print(f"✅ Created {html_file}")

# --- Create HTMLs ---
create_html(faces_df, "faces.html")
create_html(nonfaces_df, "nonfaces.html")
