import threading
import time

from utils import fetch_models, fetch_datasets, fetch_readme
from tqdm import tqdm


# Define global variables
models = None
datasets = None
readme_dict = {}


# Provide an API to get models
def get_models():
    return models


# Provide an API to get datasets
def get_datasets():
    return datasets


# Provide an API to get READMEs
def get_readme_dict():
    return readme_dict


# Start a thread to continuously update cards
def run():
    global models, datasets, readme_dict, cnt

    while True:
        try:
            new_models = fetch_models()
            new_datasets = fetch_datasets()
    
            # Add READMEs
            new_readme_dict = {}
            for model in new_models:
                new_readme_dict[model] = fetch_readme(model, "model")
    
            for dataset in new_datasets:
                new_readme_dict[dataset] = fetch_readme(dataset, "dataset")
    
            # Update global variables
            models = new_models
            datasets = new_datasets
            readme_dict = new_readme_dict
        
        except Exception as e:
            print(e)


t = threading.Thread(target=run)
t.start()
