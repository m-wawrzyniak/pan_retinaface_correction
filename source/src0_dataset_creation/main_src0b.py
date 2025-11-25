import json
from pathlib import Path

from source.src0_dataset_creation import data03_manual_labeling as data03
from source.utilities import v00_html_visuals as v00

from config import P01_extraction_config as P01


if __name__ == "__main__":
    with open(P01.RECORDINGS_INFO_PATH, "r") as f:
        recordings_info = json.load(f)
    """
    data03.manual_classify_frames_from_csv(
        recordings_info=recordings_info,
        rec_subset=P01.REC_SUBSET
    )
    """
    for rec, rec_dict in recordings_info.items():
        if rec_dict["section_start_time_ns"] != None:
            v00.export_html_paginated(
                name='manual',
                csv_path= Path(rec_dict['extraction_dir']) / 'manual_class.csv'
            )