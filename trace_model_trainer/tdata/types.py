import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import pandas as pd


class JsonSerializableMixin:
    def __json__(self):
        return dataclasses.asdict(self)


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
class TracePrediction(JsonSerializableMixin):
    source: str
    target: str
    label: Optional[int]
    score: Optional[float]

    def to_json(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "score": self.score,
            "label": self.label
        }

    def __json__(self):
        return dataclasses.asdict(self)

    def __dict__(self):
        return dataclasses.asdict(self)
