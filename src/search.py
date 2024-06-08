import gc
from itertools import product
from typing import Dict, List

import torch
from pandas import DataFrame
from tqdm import tqdm

from constants import OUTPUT_PATH
from infra.eval import eval_model
from infra.generic_trainer import generic_train
from tdata.reader import read_project
from tdata.trace_dataset import TraceDataset
from utils import get_gpu_memory_usage


def search(train_dataset_path: str, test_dataset_path: str, models: List[str], options, disable_logs: bool = False,
           **kwargs) -> DataFrame:
    train_dataset = read_project(train_dataset_path, disable_logs=disable_logs)
    test_dataset = read_project(test_dataset_path, disable_logs=disable_logs)
    entries = []

    for model_name in models:
        metrics, _ = eval_model(None, test_dataset, model_name=model_name, disable_logs=disable_logs)
        entries.append(metrics)

        for iterable_kwargs in tqdm(options, desc="Iterating through options"):
            entry = run_iteration(train_dataset, test_dataset, model_name, disable_logs=disable_logs, iterable_kwargs=iterable_kwargs,
                                  **kwargs)
            entries.append(entry)

    return DataFrame(entries)


def run_iteration(train_dataset: TraceDataset, test_dataset: TraceDataset, model_name: str, disable_logs: bool, iterable_kwargs: Dict,
                  **kwargs):
    if torch.cuda.is_available():
        print(f"GPU Memory Usage before training {model_name}: {get_gpu_memory_usage()}")

    # Train
    trained_model = generic_train(train_dataset,
                                  model_name=model_name,
                                  output_path=OUTPUT_PATH,
                                  disable_tqdm=disable_logs,
                                  **iterable_kwargs,
                                  **kwargs)

    metrics, predictions = eval_model(trained_model, test_dataset, disable_logs=disable_logs)

    # cleanup
    trained_model.to('cpu')
    del trained_model
    gc.collect()
    torch.cuda.empty_cache()

    # Display GPU memory usage after cleanup
    if torch.cuda.is_available():
        print(f"GPU Memory Usage after cleaning up {model_name}: {get_gpu_memory_usage()}")
    return {"model": model_name, **iterable_kwargs, **metrics}


def expand_dict_combinations(input_dict):
    # Extract keys and list of values
    keys = input_dict.keys()
    values = (input_dict[key] for key in keys)

    # Compute the Cartesian product of the lists of values
    combinations = product(*values)

    # Convert each combination into a dictionary
    result = [dict(zip(keys, combination)) for combination in combinations]
    return result
