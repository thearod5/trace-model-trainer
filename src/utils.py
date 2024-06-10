import gc
import inspect
import json
import os

import torch
from sklearn.preprocessing import minmax_scale


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


def get_device(disable_logs: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not disable_logs:
        print("Using device:", device)
    return device


def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Allocated memory: {allocated} GB")
    print(f"Reserved memory: {reserved} GB")


def scale(matrix):
    reshaped_matrix = matrix.reshape(-1, 1)
    # Apply MinMax scaling
    scaled_matrix = minmax_scale(reshaped_matrix)
    # Reshape back to the original matrix shape
    scaled_matrix = scaled_matrix.reshape(matrix.shape)
    return scaled_matrix


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def get_or_prompt(item_key: str, prompt: str):
    if item_key not in os.environ:
        return input(prompt)
    item_value = os.environ[item_key]
    if "PATH" in item_key:
        return os.path.expanduser(item_value)
    return item_value
