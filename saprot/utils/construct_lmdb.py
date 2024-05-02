import sys
import pandas as pd
import json

from tqdm import tqdm
from .generate_lmdb import dump_lmdb

def construct_lmdb(csv_file: str, root_dir: str, dataset_name: str, task_type: str) -> None:
    """
    Construct LMDB dataset from CSV file
    Args:
        csv_file:  Path to CSV file

        root_dir:  Root directory to save LMDB dataset

        dataset_name: Name of the dataset

        task_type: Type of the task
    """

    assert task_type in ["classification", "regression", "token_classification"]

    # Load CSV file
    df = pd.read_csv(csv_file)
    if task_type == "token_classification":
        for index, value in df["label"].items():
            df.loc[index, "label"] = [int(item.strip()) for item in value.split(",")]
        
    # Construct data dictionary
    data_dicts = {
        "train": {},
        "valid": {},
        "test": {}
    }

    label_keys = {
        "classification": "label",
        "token_classification": "label",
        "regression": "fitness",
    }

    # Go through each row of the CSV file
    for i, row in tqdm(df.iterrows()):
        seq, label, stage = row
        tmp_dict = data_dicts[stage]

        # Add data to the dictionary
        sample = {
            "seq": seq,
            label_keys[task_type]: label
        }
        tmp_dict[len(tmp_dict)] = json.dumps(sample)

    for stage in ["train", "valid", "test"]:
        tmp_dict = data_dicts[stage]
        tmp_dict["length"] = len(tmp_dict)

        lmdb_dir = f"{root_dir}/{dataset_name}/{stage}"
        dump_lmdb(tmp_dict, lmdb_dir)