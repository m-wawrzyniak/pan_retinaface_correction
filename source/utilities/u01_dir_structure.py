import os
import json
from pprint import pprint

from config import P01_extraction_config as P01

def create_project_skeleton(root, classifier_name, dataset_name, project_json_path):
    """
    Create directory skeleton for classifier + dataset based only on:
    - ROOT
    - project_dir_structure.json
    - CLASSIFIER_NAME
    - DATASET_NAME

    Reads classifier + dataset templates from the 'config' section of project spec.
    """

    # ------------------------------------------------------------------
    # Load project structure
    # ------------------------------------------------------------------
    with open(project_json_path, "r") as f:
        project_spec = json.load(f)

    # Extract template paths from project spec
    config_section = project_spec["config"]
    classifier_template_path = os.path.join(root, config_section["classifier_dir_structure.json"])
    dataset_template_path = os.path.join(root, config_section["dataset_dir_structure.json"])

    # Load classifier/dataset template JSONs
    with open(classifier_template_path, "r") as f:
        classifier_spec = json.load(f)

    with open(dataset_template_path, "r") as f:
        dataset_spec = json.load(f)

    # Extract base paths inside /data directory
    data_section = project_spec["data"]
    classifiers_base = os.path.join(root, data_section["classifiers"]["_dir"], classifier_name)
    datasets_base = os.path.join(root, data_section["datasets"]["_dir"], dataset_name)

    # ------------------------------------------------------------------
    # Helper to create directories based on `_dir` entries
    # ------------------------------------------------------------------
    def create_dirs(base_path, spec):
        for key, val in spec.items():

            if isinstance(val, dict):

                # Directory definition
                if "_dir" in val:
                    dir_path = os.path.join(base_path, val["_dir"])
                    if os.path.exists(dir_path):
                        print(f"[WARN] Exist: {dir_path}")
                    else:
                        print(f"[CREATE] Dir: {dir_path}")
                        os.makedirs(dir_path, exist_ok=True)

                    # Continue recursively
                    create_dirs(base_path, val)

                else:
                    # Just a nested object
                    create_dirs(base_path, val)

            # File — ignore
            elif isinstance(val, str):
                continue

    # ------------------------------------------------------------------
    # Create classifier tree
    # ------------------------------------------------------------------
    print("\n=== CLASSIFIER SETUP ===")

    if os.path.exists(classifiers_base):
        print(f"[WARN] Classifier dir exists: {classifiers_base}")
    else:
        print(f"[CREATE] Classifier root: {classifiers_base}")
        os.makedirs(classifiers_base, exist_ok=True)

    create_dirs(classifiers_base, classifier_spec)

    # ------------------------------------------------------------------
    # Create dataset tree
    # ------------------------------------------------------------------
    print("\n=== DATASET SETUP ===")

    if os.path.exists(datasets_base):
        print(f"[WARN] Dataset dir exists: {datasets_base}")
    else:
        print(f"[CREATE] Dataset root: {datasets_base}")
        os.makedirs(datasets_base, exist_ok=True)

    create_dirs(datasets_base, dataset_spec)

    print("\n=== DONE ===")

def build_absolute_paths(root, classifier_name, dataset_name, project_json_path):
    """
    Builds a dictionary containing the ABSOLUTE PATHS for the entire project,
    including dataset and classifier trees, resolved for the given names.
    """

    # ------------------------------------------------------------------
    # Load project structure
    # ------------------------------------------------------------------
    with open(project_json_path, "r") as f:
        project_spec = json.load(f)

    # Extract config file paths
    config = project_spec["config"]
    classifier_template_path = os.path.join(root, config["classifier_dir_structure.json"])
    dataset_template_path = os.path.join(root, config["dataset_dir_structure.json"])

    # Load templates
    with open(classifier_template_path, "r") as f:
        classifier_template = json.load(f)

    with open(dataset_template_path, "r") as f:
        dataset_template = json.load(f)

    # Extract base dataset/classifier directories
    data_spec = project_spec["data"]
    classifiers_base = os.path.join(root, data_spec["classifiers"]["_dir"], classifier_name)
    datasets_base = os.path.join(root, data_spec["datasets"]["_dir"], dataset_name)

    # ------------------------------------------------------------------
    # Recursive resolver
    # ------------------------------------------------------------------
    def resolve(base_path, tree):
        resolved = {}

        for key, val in tree.items():

            # Directory
            if isinstance(val, dict):
                if "_dir" in val:
                    abs_dir = os.path.join(base_path, val["_dir"])
                    resolved[key] = {"_dir": abs_dir}

                    # Resolve nested items under same base
                    nested = {k: v for k, v in val.items() if k != "_dir"}
                    for nk, nv in resolve(base_path, nested).items():
                        resolved[key][nk] = nv

                else:
                    # Nested structure without its own base
                    resolved[key] = resolve(base_path, val)

            # File
            elif isinstance(val, str):
                resolved[key] = os.path.join(base_path, val)

            else:
                raise ValueError(f"Unsupported type in JSON: {type(val)}")

        return resolved

    # ------------------------------------------------------------------
    # Build final master dictionary
    # ------------------------------------------------------------------
    result = {
        "ROOT": root,
        "config": {
            "project_dir_structure": os.path.abspath(project_json_path),
            "dataset_structure": os.path.abspath(dataset_template_path),
            "classifier_structure": os.path.abspath(classifier_template_path)
        },
        "data": {
            "CLASSIFIER": resolve(classifiers_base, classifier_template),
            "DATASET": resolve(datasets_base, dataset_template)
        }
    }

    return result

if __name__ == "__main__":
    """
    create_project_skeleton(
        root=P01.ROOT,
        classifier_name="class_v00",
        dataset_name="dataset_v00",
        project_json_path=P01.PROJECT_STRUCT
    )
    """
    paths_dict = build_absolute_paths(
        root=P01.ROOT,
        classifier_name="class_v00",
        dataset_name="dataset_v00",
        project_json_path=P01.PROJECT_STRUCT
    )
    pprint(paths_dict)