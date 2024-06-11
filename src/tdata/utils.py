from typing import Set

import numpy as np
from pandas import DataFrame

from tdata.types import Artifact


def get_artifact_ids_from_trace_df(trace_df: DataFrame) -> Set:
    artifact_ids = set()

    for i, row in trace_df.iterrows():
        artifact_ids.add(row['source'])
        artifact_ids.add(row['target'])
    return artifact_ids


def get_content(a: Artifact):
    """
    Returns summary of artifacts if exists otherwise returns its content.
    :param a: Artifact to extract content for.
    :return: Artifact content.
    """
    if "summary" not in a:
        return a["content"]
    s = a["summary"]
    if s is None:
        return a["content"]
    return a["content"] if isinstance(s, float) and (np.isnan(s) or s is None) else s
