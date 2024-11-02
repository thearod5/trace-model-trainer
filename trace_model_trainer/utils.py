import gc
import inspect
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List

import torch
from pandas import DataFrame
from sklearn.preprocessing import minmax_scale

from trace_model_trainer.tdata.types import TracePrediction


def read_json(f):
    with open(f) as file_pointer:
        return json.load(file_pointer)


def write_json(content: dict, save_path: str, pretty: bool = False):
    with open(save_path, 'w') as file_pointer:
        json.dump(content, fp=file_pointer, default=lambda o: o.__json__(), indent=4 if pretty else None)


def t_id_creator(trace_row=None, source=None, target=None):
    """
    Creates trace id for row in trace file.
    :param trace_row: Row in trace data frame.
    :return: Trace ID.
    """
    if trace_row is not None:
        source = trace_row['source']
        target = trace_row['target']

    return f"{source}*{target}"


def has_param(f, param):
    """
    Check if the function `f` accepts a parameter named `param`.

    Args:
    f (function): The function to check.
    param (str): The name of the parameter to look for.

    Returns:
    bool: True if `f` accepts a parameter named `param`, False otherwise.
    """
    try:
        # Get the signature of the function
        sig = inspect.signature(f)
        # Check if the parameter is in the function's parameters
        return param in sig.parameters
    except ValueError:
        # If `f` is not a callable, return False
        return False


def scale(matrix):
    reshaped_matrix = matrix.reshape(-1, 1)
    # Apply MinMax scaling
    scaled_matrix = minmax_scale(reshaped_matrix)
    # Reshape back to the original matrix shape
    scaled_matrix = scaled_matrix.reshape(matrix.shape)
    return scaled_matrix


def create_trace_map(trace_df: DataFrame, group_attr="source", target_attr="target"):
    """
    Creates map of source ids to traced target ids.
    :param trace_df: The data frame containing trace links.
    :param group_attr: The attribute to group each row by.
    :param target_attr: The corresponding attribute to attach to each group.
    :return: Map of sources to traced targets.
    """
    assert "label" in trace_df.columns, f"Result: {trace_df.columns}"
    source2target = defaultdict(dict)

    # Filter rows where label is 1
    filtered_df = trace_df[trace_df["label"] == 1]

    # Iterate using itertuples for faster row access
    for row in filtered_df.itertuples(index=False):
        group_val = getattr(row, group_attr)
        target_val = getattr(row, target_attr)
        source2target[group_val][target_val] = 1

    return source2target


def create_predictions_from_matrix(sources: List[str], targets: List[str], similarity_matrix: List[List[float]]) -> (
        List)[TracePrediction]:
    predictions = []
    for s_index, source in enumerate(sources):
        for t_index, target in enumerate(targets):
            score = similarity_matrix[s_index][t_index]
            prediction = TracePrediction(source=source, target=target, label=None, score=score)
            predictions.append(prediction)
    return predictions


def group_by(items: List[Dict], group_key: str) -> Dict[str, List[Dict]]:
    is_dict = isinstance(items[0], dict)
    group2items = defaultdict(list)
    for item in items:
        group = item[group_key] if is_dict else getattr(item, group_key)
        group2items[group].append(item)
    return group2items


def clear_dir(dir_path: str) -> None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


"""
Memory Utils
"""


def get_device(disable_logs: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not disable_logs:
        print("Using device:", device)
    return device


def print_gpu_memory():
    if not torch.cuda.is_available():
        print("No GPU available.")
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Allocated memory: {allocated} GB")
    print(f"Reserved memory: {reserved} GB")


def clear_memory(model=None):
    if model:
        model.to("cpu")
        del model
    gc.collect()
    torch.cuda.empty_cache()
