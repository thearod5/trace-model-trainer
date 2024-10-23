import os.path
from typing import Literal

from datasets import Dataset, load_dataset
from pandas import DataFrame

from trace_model_trainer.tdata.reader import read_project
from trace_model_trainer.tdata.trace_dataset import TraceDataset

DatasetTypes = Literal["train", "traceability"]


def load_traceability_dataset(dataset_name: str, **kwargs) -> TraceDataset:
    """
    Loads traceability training dataset
    :param dataset_name: The name of dataset or path to it.
    :return: The dataset requested.
    """
    if os.path.exists(dataset_name):
        return read_project(dataset_name)
    artifacts = load_dataset(dataset_name, "artifacts", **kwargs)
    traces = load_dataset(dataset_name, "traces", **kwargs)
    matrices = load_dataset(dataset_name, "matrices", **kwargs)

    artifacts = artifacts["train"] if "train" in artifacts else artifacts["artifacts"]
    traces = traces["train"] if "train" in traces else traces["traces"]
    matrices = matrices["train"] if "train" in matrices else matrices["matrices"]

    dataset = TraceDataset(
        artifact_df=_to_artifact_df(artifacts),
        trace_df=_to_trace_df(traces),
        layer_df=matrices.to_pandas()
    )
    return dataset


def _to_trace_df(dataset: Dataset) -> DataFrame:
    if "s_id" in dataset.column_names:
        df = dataset.rename_columns({"s_id": "source", "t_id": "target"})
    df = dataset.to_pandas()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    return df


def _to_artifact_df(dataset: Dataset) -> DataFrame:
    df = dataset.to_pandas()
    df["id"] = df["id"].astype(str)
    return df
