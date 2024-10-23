import os
from typing import Dict, Iterable

import pandas as pd
from pandas import DataFrame

from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.utils import has_param, read_json


def read_project(project_path: str, disable_logs: bool = False) -> TraceDataset:
    if not os.path.exists(project_path):
        raise Exception(f"Project path does not contain files:{project_path}")
    project_path = os.path.abspath(project_path)

    if all([os.path.exists(os.path.join(project_path, f)) for f in ["artifacts.csv", "traces.csv", "matrices.csv"]]):
        return TraceDataset(
            artifact_df=pd.read_csv(os.path.join(project_path, "artifacts.csv")),
            trace_df=pd.read_csv(os.path.join(project_path, "traces.csv")),
            layer_df=pd.read_csv(os.path.join(project_path, "matrices.csv"))
        )

    tim = read_tim(project_path)

    file2layer = {a['fileName']: a['type'] for a in tim["artifacts"]}
    file2traced_layers = {f['fileName']: (f['sourceType'], f['targetType']) for f in tim["traces"]}
    layer_df = DataFrame(file2traced_layers.values())

    artifact_df = create_artifact_df(project_path, file2layer)
    trace_df = create_trace_df(project_path, file2traced_layers.keys())

    if not disable_logs:
        print("Artifacts:", len(artifact_df))
        print("Traces:", len(trace_df))
    return TraceDataset(artifact_df, trace_df, layer_df)


def read_safa_project(project_path):
    tim = read_tim(project_path)
    file2layer = {a['fileName']: a['type'] for a in tim["artifacts"]}
    file2traced_layers = {f['fileName']: (f['sourceType'], f['targetType']) for f in tim["traces"]}
    layer_df = DataFrame(file2traced_layers.values())
    artifact_df = create_artifact_df(project_path, file2layer)
    trace_df = create_trace_df(project_path, file2traced_layers.keys())
    return artifact_df, layer_df, trace_df


def create_artifact_df(project_path: str, file2layer: Dict[str, str]):
    dfs = []
    for f in file2layer.keys():
        df = pd.read_csv(os.path.join(project_path, f))
        if "layer" not in df.columns:
            df["layer"] = file2layer[f]
        dfs.append(df)
    return pd.concat(dfs)


def create_trace_df(project_path: str, files: Iterable[str]):
    """
    Reads traces in project found in files.
    :param project_path: Path to folder containing files.
    :param files: List of trace files to read.
    :return: TraceDataFrame
    """
    dfs = []
    for f in files:
        file_path = os.path.join(project_path, f)
        f_df = read_file(file_path, k="traces")
        if ".json" in f:
            f_df = f_df[f_df['traceType'] == "MANUAL"]
            f_df = f_df.rename({"sourceName": "source", "targetName": "target"}, axis=1)
            f_df = f_df.drop("traceType", axis=1)
        f_df['label'] = 1
        dfs.append(f_df)

    df = pd.concat(dfs)
    return df


def read_tim(project_path: str) -> dict:
    """
    Reads project tim file.
    :param project_path: Path to project.
    :return: Tim file content as dictionary.
    """
    tim_path = os.path.join(project_path, "tim.json")
    if not os.path.exists(tim_path):
        raise Exception(f"Expected tim.json file to be defined at {project_path}")
    tim = read_json(tim_path)
    return tim


def extract_trace_layers_from_file(f):
    """
    Extracts the layers being traced from file name.

    :param f:
    :return:
    """

    f = f[:f.index(".")]
    return f.split("2")


def read_file(f, **kwargs) -> pd.DataFrame:
    """
    Reads CSV or JSON file.
    :param f: Path to file.
    :param kwargs: Kwargs passed to reader function.
    :return: DataFrame with entities.
    """
    file_readers = {
        "csv": csv_reader,
        "json": json_reader
    }

    ext = os.path.splitext(f)[1][1:]
    reader_func = file_readers[ext]

    kwargs = {k: v for k, v in kwargs.items() if has_param(reader_func, k)}
    return reader_func(f, **kwargs)


def csv_reader(f):
    """
    Reads csv file.
    :param f: Path to file.
    :return: DataFrame.
    """
    return pd.read_csv(f)


def json_reader(file_path, k: str):
    """
    Reads json file whose entries are stored under key.
    :param file_path: Path to JSON file.
    :param k: Key to retrieve entities under.
    :return: Return DataFrame containing entities.
    """
    f_json = read_json(file_path)
    return pd.DataFrame(f_json[k])
