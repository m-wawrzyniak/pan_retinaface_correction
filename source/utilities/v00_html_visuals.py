import os
import pandas as pd
from pathlib import Path
import math

def present_html_class(csv_path, html_file_name):
    """
    Create an HTML visualization of manually classified frames.
    The HTML shows two sections: face and non-face.
    The JPGs must be in the same directory as the CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to manual_class.csv
    """

    csv_path = Path(csv_path)
    base_dir = csv_path.parent
    title = base_dir.parent.name  # directory name becomes title

    df = pd.read_csv(csv_path)

    # Separate into two groups
    face_df = df[df["is_face"] == 1]
    nonface_df = df[df["is_face"] == 0]

    html_path = base_dir / html_file_name

    def block_for(df_subset, label):
        """Return HTML block with embedded images."""
        html = f"<h2>{label}</h2>\n<div style='display:flex; flex-wrap:wrap;'>\n"
        for fname in df_subset["frame"]:
            img_path = base_dir / fname
            if not img_path.exists():
                continue
            html += (
                f"<div style='margin:10px;'>"
                f"<img src='{fname}' style='max-width:320px; height:auto; display:block;'>"
                f"<p style='text-align:center;'>{fname}</p>"
                f"</div>"
            )
        html += "</div>\n"
        return html

    # Build full HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title} — Manual Classification</title>
</head>
<body style="font-family: sans-serif;">
    <h1>Manual Classification Results for: {title}</h1>

    {block_for(face_df, "Faces")}
    <hr>
    {block_for(nonface_df, "Non-Faces")}

</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    print(f"✅ HTML visualization created at: {html_path}")

def export_html_paginated(csv_path, page_size=100):
    """
    Creates paginated HTML visualization for is_face / non_face images.

    Parameters
    ----------
    csv_path : str or Path
        Path to manual_class.csv
    page_size : int
        How many images per HTML page (default 100)
    """

    csv_path = Path(csv_path)
    base_dir = csv_path.parent
    title = base_dir.name

    df = pd.read_csv(csv_path)
    face_df = df[df["is_face"] == 1].reset_index(drop=True)
    nonface_df = df[df["is_face"] == 0].reset_index(drop=True)

    def write_page(df_subset, label, page_idx, total_pages):
        """Write a single paginated HTML file."""
        file_name = f"{label.lower()}_page_{page_idx+1}.html"
        out_path = base_dir / file_name

        start = page_idx * page_size
        end = min(start + page_size, len(df_subset))
        subset = df_subset.iloc[start:end]

        # Navigation links
        nav = "<div>"
        if page_idx > 0:
            nav += f"<a href='{label.lower()}_page_{page_idx}.html'>⬅️ Prev</a> | "
        nav += f"Page {page_idx+1} / {total_pages}"
        if page_idx < total_pages - 1:
            nav += f" | <a href='{label.lower()}_page_{page_idx+2}.html'>Next ➡️</a>"
        nav += "</div><hr>"

        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title} — {label} {page_idx+1}/{total_pages}</title>
</head>
<body style="font-family:sans-serif;">
<h1>{label} — {title}</h1>
{nav}
<div style='display:flex; flex-wrap:wrap;'>
"""

        for fname in subset["frame"]:
            img_path = base_dir / fname
            if not img_path.exists():
                continue

            html += (
                f"<div style='margin:10px;'>"
                f"<img loading='lazy' src='{fname}' "
                f"style='max-width:320px;height:auto;display:block;'>"
                f"<p style='text-align:center;'>{fname}</p>"
                f"</div>"
            )

        html += f"</div>{nav}</body></html>"

        out_path.write_text(html, encoding="utf-8")
        print(f"📄 Wrote {out_path.name}")

    def paginated_export(df_subset, label):
        if len(df_subset) == 0:
            return []

        total_pages = math.ceil(len(df_subset) / page_size)
        for page_idx in range(total_pages):
            write_page(df_subset, label, page_idx, total_pages)

        return [
            f"{label.lower()}_page_{i+1}.html"
            for i in range(total_pages)
        ]

    # Generate pages
    face_pages = paginated_export(face_df, "Faces")
    nonface_pages = paginated_export(nonface_df, "NonFaces")

    # Create index.html as a hub
    index_path = base_dir / "index.html"
    html_index = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title} — Overview</title>
</head>
<body style="font-family:sans-serif;">
<h1>Manual Classification Overview — {title}</h1>

<h2>Faces ({len(face_df)})</h2>
<ul>
    {''.join(f"<li><a href='{p}'>{p}</a></li>" for p in face_pages)}
</ul>

<h2>Non-Faces ({len(nonface_df)})</h2>
<ul>
    {''.join(f"<li><a href='{p}'>{p}</a></li>" for p in nonface_pages)}
</ul>

</body>
</html>
"""
    index_path.write_text(html_index, encoding="utf-8")
    print(f"✅ Created paginated viewer at {index_path}")
