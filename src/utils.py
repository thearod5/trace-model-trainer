import inspect
import json

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


def get_gpu_memory_usage():
    """Return the current GPU memory usage in MB."""
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated()
    return allocated / 1024 ** 2  # convert from Bytes to MB


def scale(matrix):
    reshaped_matrix = matrix.reshape(-1, 1)
    # Apply MinMax scaling
    scaled_matrix = minmax_scale(reshaped_matrix)
    # Reshape back to the original matrix shape
    scaled_matrix = scaled_matrix.reshape(matrix.shape)
    return scaled_matrix
