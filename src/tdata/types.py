from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import pandas as pd


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


@dataclass
class TracePrediction:
    source: str
    target: str
    label: int
    score: float
