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

    assert task_type in ["classification", "regression", "token_classification", "pair_regression", "pair_classification"]

    # Load CSV file
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.lower()
    if task_type == "token_classification":
        for index, row in df.iterrows():
            if row["stage"] == "train":
                df.at[index, "label"] = [int(item.strip()) for item in row["label"].split(",")][:1024]
            else:
                df.at[index, "label"] = [int(item.strip()) for item in row["label"].split(",")]
        
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
        "pair_regression": "label",
        "pair_classification": "label"
    }

    if task_type in ["pair_regression", "pair_classification"]:
        # Go through each row of the CSV file
        for i, row in tqdm(df.iterrows(), total=len(df)):
            # seq, label, stage = row
            label = row["label"]
            stage = row["stage"]
            name_1 = row["name_1"]
            name_2 = row["name_2"]
            chain_1 = row["chain_1"]
            chain_2 = row["chain_2"]

            if stage == "train":
                seq_1 = row["sequence_1"][:2048]
                seq_2 = row["sequence_2"][:2048]
            else:
                seq_1 = row["sequence_1"]
                seq_2 = row["sequence_2"]

            tmp_dict = data_dicts[stage]

            # Add data to the dictionary
            sample = {
                "seq_1": seq_1,
                "seq_2": seq_2,
                "name_1": name_1, 
                "name_2": name_2, 
                "chain_1": chain_1,
                "chain_2": chain_2,
                label_keys[task_type]: label
            }
            tmp_dict[len(tmp_dict)] = json.dumps(sample)
        
    else:
        # Go through each row of the CSV file
        for i, row in tqdm(df.iterrows(), total=len(df)):
            # seq, label, stage = row
            label = row["label"]
            stage = row["stage"]

            if stage == "train":
                seq = row["sequence"][:2048]
            else:
                seq = row["sequence"]

            tmp_dict = data_dicts[stage]

            # Add data to the dictionary
            sample = {
                "seq": seq,
                label_keys[task_type]: label
            }
            tmp_dict[len(tmp_dict)] = json.dumps(sample)

    # If the task is a classification task, check the validity of the labels
    if "classification" in task_type:
        # Go through each row of the CSV file
        for i, row in tqdm(df.iterrows(), total=len(df)):
            label = row["label"]
            if type(label) == list:
                for item in label:
                    assert type(item) == int, "Labels for classification task must be integer."

            else:
                assert type(label) == int, "Labels for classification task must be integer."

    for stage in ["train", "valid", "test"]:
        tmp_dict = data_dicts[stage]
        tmp_dict["length"] = len(tmp_dict)

        lmdb_dir = f"{root_dir}/{dataset_name}/{stage}"
        dump_lmdb(tmp_dict, lmdb_dir)