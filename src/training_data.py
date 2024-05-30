from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from reader import ProjectData
from trace_dataset import TraceDataset
from utils import t_id_creator

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


def create_training_data(project_data: ProjectData) -> TrainingData:
    train_df, test_df = create_splits(project_data)
    train_df = create_triplet_df(train_df, project_data.artifact_map)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    print(f"Train({len(test_df)}) Val({len(val_df)}) Test({len(test_df)})")
    return TrainingData(train_df=train_df,
                        val_df=val_df, test_dataset=TraceDataset(test_df, project_data.artifact_map),
                        project_data=project_data)


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


def create_entries(s_artifacts, t_artifacts, trace_map):
    training_entries = []
    for i, s_a in s_artifacts.iterrows():
        for j, t_a in t_artifacts.iterrows():
            s_id = s_a["id"]
            t_id = t_a["id"]
            t_id = t_id_creator({'source': s_id, 'target': t_id})
            training_entries.append({
                "s_id": s_id,
                "t_id": t_a["id"],
                "s_layer": s_a["layer"],
                "t_layer": t_a["layer"],
                "label": 1 if t_id in trace_map else 0
            })
    return training_entries


def create_triplet_df(df, artifact_map: Dict[str, str]):
    # Split data into positive and negative pairs
    positive_pairs = df[df['label'] == 1].copy()
    negative_pairs = df[df['label'] == 0].copy()

    # Create triplets
    triplets = []
    for _, pos_row in positive_pairs.iterrows():
        anchor_id = pos_row['s_id']
        positive_id = pos_row['t_id']

        # Find a negative example
        negative_candidates = negative_pairs[negative_pairs['s_id'] == anchor_id]
        if not negative_candidates.empty:
            negative_id = negative_candidates.sample(1).iloc[0]['t_id']
            triplets.append((anchor_id, positive_id, negative_id))

    triplets = [(artifact_map[a], artifact_map[b], artifact_map[c]) for a, b, c in triplets]
    triplet_df = pd.DataFrame(triplets, columns=['anchor', 'positive', 'negative'])
    return triplet_df
