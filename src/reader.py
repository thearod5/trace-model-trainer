import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd

from utils import has_param, read_json, t_id_creator

""""
ArtifactDataFrame: (id, content, summary)
TraceDataFrame: (source,target,label)

TODO: TIM file is not being read.
"""


@dataclass
class ProjectData:
    artifact_df: pd.DataFrame
    trace_df: pd.DataFrame
    artifact_map: Dict[str, str]
    trace_map: Dict[str, Dict]
    trace_layers: List[Tuple[str, str]]


class Artifact(TypedDict):
    id: str
    content: str
    summary: Optional[str]


def read_project(project_path: str):
    project_path = os.path.abspath(project_path)

    tim = read_tim(project_path)

    file2layer = {a['fileName']: a['type'] for a in tim["artifacts"]}
    file2traces = {f['fileName']: (f['sourceType'], f['targetType']) for f in tim["traces"]}

    artifact_df = create_artifact_df(project_path, file2layer)
    artifact_map = {a['id']: get_content(a.to_dict()) for i, a in artifact_df.iterrows()}

    trace_df = create_trace_df(project_path, file2traces.keys())

    trace_map: Dict[str, Dict] = {t_id_creator(r): r.to_dict() for i, r in trace_df.iterrows()}

    print("Artifacts:", len(artifact_df))
    print("Traces:", len(trace_df))

    return ProjectData(artifact_df=artifact_df,
                       trace_df=trace_df,
                       artifact_map=artifact_map,
                       trace_map=trace_map,
                       trace_layers=list(file2traces.values()))


def create_artifact_df(project_path: str, file2layer: Dict[str, str]):
    dfs = []
    for f in file2layer.keys():
        df = pd.read_csv(os.path.join(project_path, f))
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
        if ".json" in files:
            f_df = f_df[f_df['traceType'] == "MANUAL"]
            f_df = f_df.rename({"sourceName": "source", "targetName": "target"}, axis=1)
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


def get_content(a: Artifact):
    """
    Returns summary of artifacts if exists otherwise returns its content.
    :param a: Artifact to extract content for.
    :return: Artifact content.
    """
    if "summary" not in a:
        return a["content"]
    s = a["summary"]
    return a["content"] if isinstance(s, float) and np.isnan(s) else s


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
