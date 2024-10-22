import gc
import inspect
import json
from collections import defaultdict
from typing import List

import torch
from pandas import DataFrame
from sklearn.preprocessing import minmax_scale

from trace_model_trainer.readers.types import TracePrediction


def read_json(f):
    with open(f) as file_pointer:
        return json.load(file_pointer)


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


def create_source2targets(trace_df: DataFrame):
    source2target = defaultdict(dict)
    for _, t in trace_df.iterrows():
        label = t.get("label", 1)
        if label == 1:
            source2target[t["source"]][t["target"]] = 1
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
