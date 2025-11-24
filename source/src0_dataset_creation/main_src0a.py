import json

from source.src0_dataset_creation import data00_rec_info as data00, data01_extract_frames as data01, data02_deduplicate_frames as data02

from config import P01_extraction_config as P01

if __name__ == "__main__":
    print('\n00. Extracting recording info..')
    data00.build_recordings_info(P01.TIMESERIES_DATA, P01.SECTIONS_CSV, section_name="manipulative.begin")

    print('\n01. Extracting face frames..')
    data01.extract_faces_for_all(P01.FACE_MAPPER_DIR, P01.RECORDINGS_INFO_PATH, subset_ids=P01.REC_SUBSET)

    print('\n02. Deduplicating face frames..')
    with open(P01.RECORDINGS_INFO_PATH, "r") as f:
        recordings_info = json.load(f)
    data02.deduplicate_face_frames_csv(
        recordings_info=recordings_info,
        rec_subset=P01.REC_SUBSET,
        threshold=P01.DEDUP_THRESHOLD
    )
