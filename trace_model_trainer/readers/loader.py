import os.path
from typing import Literal

from datasets import Dataset, load_dataset
from pandas import DataFrame

from readers.reader import read_project
from readers.trace_dataset import TraceDataset

DatasetTypes = Literal["train", "traceability"]


def load_traceability_dataset(dataset_name: str) -> TraceDataset:
    """
    Loads traceability training dataset
    :param dataset_name: The name of dataset or path to it.
    :return: The dataset requested.
    """
    if os.path.exists(dataset_name):
        return read_project(dataset_name)
    artifacts = load_dataset(dataset_name, "artifacts")
    traces = load_dataset(dataset_name, "traces")
    matrices = load_dataset(dataset_name, "matrices")

    dataset = TraceDataset(
        artifact_df=_to_artifact_df(artifacts["train"]),
        trace_df=_to_trace_df(traces["train"]),
        layer_df=matrices["train"].to_pandas()
    )
    return dataset


def _to_trace_df(dataset: Dataset) -> DataFrame:
    df = dataset.rename_columns({"s_id": "source", "t_id": "target"}).to_pandas()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    return df


def _to_artifact_df(dataset: Dataset) -> DataFrame:
    df = dataset.to_pandas()
    df["id"] = df["id"].astype(str)
    return df
