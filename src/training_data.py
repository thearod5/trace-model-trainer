from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from tdata.trace_dataset import TraceDataset

"""

TrainingData: (anchor_id, positive_id, negative_id)
ValData: TraceDataset
TestData: TraceDataset

TraceDataset: (source, target, label)

"""


@dataclass
class TrainingData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_dataset: TraceDataset
    project_data: ProjectData


def create_splits(project_data: ProjectData):
    artifact_df = project_data.artifact_df
    trace_layers = project_data.trace_layers
    trace_map = project_data.trace_map

    training_entries = []
    test_entries = []

    for s_type, t_type in trace_layers:
        print(f"Tracing {s_type} -> {t_type}")
        source_artifacts = artifact_df[artifact_df["layer"] == s_type]
        t_artifacts = artifact_df[artifact_df["layer"] == t_type]

        train_targets, test_targets = train_test_split(t_artifacts, test_size=0.1, random_state=42)

        training_entries.extend(create_entries(source_artifacts, train_targets, trace_map))
        test_entries.extend(create_entries(source_artifacts, test_targets, trace_map))

    return pd.DataFrame(training_entries), pd.DataFrame(test_entries)
