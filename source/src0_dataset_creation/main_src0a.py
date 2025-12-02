import json

from source.src0_dataset_creation import (data00_rec_info as data00, data01_extract_frames as data01,
                                          data02_deduplicate_frames as data02, exp_data02)
from source.utilities import u01_dir_structure as u01

from config import (P01_extraction_config as P01, P02_model_config as P02)

if __name__ == "__main__":
    paths_dict = u01.build_absolute_paths(
        root=P01.ROOT,
        classifier_name=P01.CLASSIFIER_NAME,
        dataset_name=P01.DATASET_NAME,
        project_json_path=P01.PROJECT_STRUCT
    )

    print('\n00. Extracting recording info..')
    data00.build_recordings_info(P01.TIMESERIES_DATA, P01.SECTIONS_CSV,
                                 section_name="manipulative.begin",
                                 json_path=paths_dict['data']['datasets'][P01.DATASET_NAME]['recordings_info.json'],
                                 extraction_root=paths_dict['data']['datasets'][P01.DATASET_NAME]['extracted_frames']['_dir'],
                                 model_csv_root=paths_dict['data']['datasets'][P01.DATASET_NAME]['model_classification']['_dir'],
                                 manual_csv_root=paths_dict['data']['datasets'][P01.DATASET_NAME]['manual_classification']['_dir'],
                                 rec_subset=P01.REC_SUBSET)

    print('\n01. Extracting face frames..')
    data01.extract_faces_for_all(P01.FACE_MAPPER_DIR,
                                 recordings_info_path=paths_dict['data']['datasets'][P01.DATASET_NAME]['recordings_info.json'],
                                 subset_ids=P01.REC_SUBSET)


    print('\n02. Deduplicating face frames..')
    with open(paths_dict['data']['datasets'][P01.DATASET_NAME]['recordings_info.json'], "r") as f:
        recordings_info = json.load(f)
    """
    data02.deduplicate_face_frames_csv(
        recordings_info=recordings_info,
        rec_subset=P01.REC_SUBSET,
        threshold=P01.DEDUP_THRESHOLD
    )
    """
    exp_data02.sample_face_frames_csv(
        recordings_info=recordings_info,
        rec_subset=P01.REC_SUBSET,
        threshold=P01.DEDUP_THRESHOLD,
        n_clusters=P01.N_CLUSTERS,
        min_per_cluster=P01.MIN_PER_CLUSTER,
        max_total=P01.TARGET_FRAMES_PER_REC,
        seed=P02.SEED
    )
