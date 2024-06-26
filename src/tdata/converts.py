from collections import defaultdict

import pandas as pd
from datasets import Dataset
from sentence_transformers.losses import ContrastiveTensionDataLoader

from tdata.trace_dataset import TraceDataset
from utils import t_id_creator


def to_dataset(dataset: TraceDataset, dataset_name: str):
    if dataset_name == "mnrl":
        return to_mnrl_dataset(dataset)
    elif dataset_name == "float":
        return to_float_dataset(dataset)
    elif dataset_name == "triplet":
        return to_triplet_dataset(dataset)
    elif dataset_name == "contrastive_tension":
        return to_contrastive_tension_dataset(dataset)
    else:
        raise Exception(f"Unknown dataset type: {dataset_name}")


def to_contrastive_tension_dataset(dataset: TraceDataset):
    train_examples = list(dataset.artifact_df["content"])
    examples = [e for e in ContrastiveTensionDataLoader(train_examples, batch_size=6, pos_neg_ratio=3)]

    text1 = [e.texts[0] for e_batch in examples for e in e_batch]
    text2 = [e.texts[1] for e_batch in examples for e in e_batch]
    labels = [e.label for e_batch in examples for e in e_batch]
    return Dataset.from_dict({
        "text1": text1,
        "texts2": text2,
        "label": labels
    })


def to_float_dataset(dataset: TraceDataset):
    sources = []
    targets = []
    labels = []
    t_map = dataset.trace_map

    for source_ids, target_ids in dataset.get_layer_iterator():
        for s_id in source_ids:
            for t_id in target_ids:
                trace_id = t_id_creator(source=s_id, target=t_id)
                sources.append(dataset.artifact_map[s_id])
                targets.append(dataset.artifact_map[t_id])
                labels.append(1 if trace_id in t_map else 0)

    return Dataset.from_dict({
        "sentence1": sources,
        "sentence2": targets,
        "label": labels
    })


def to_mnrl_dataset(dataset: TraceDataset):
    anchors = []
    positives = []
    for i, row in dataset.trace_df.iterrows():
        s_id = row['source']
        t_id = row['target']
        anchors.append(dataset.artifact_map[s_id])
        positives.append(dataset.artifact_map[t_id])

    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives
    })


def to_triplet_dataset(dataset: TraceDataset):
    artifact_map = dataset.artifact_map

    triplet_lookup = create_triplet_lookup(dataset)

    # Create triplets
    anchors = []
    positives = []
    negatives = []

    for source_id, lookup in triplet_lookup.items():
        positive_links = pd.Series(lookup['pos'])
        negative_links = pd.Series(lookup['neg']).sample(frac=1)
        for positive_id, negative_id in zip(positive_links, negative_links):
            anchors.append(artifact_map[source_id])
            positives.append(artifact_map[positive_id])
            negatives.append(artifact_map[negative_id])

    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives
    })


def create_triplet_lookup(dataset: TraceDataset):
    lookup = defaultdict(lambda: {"pos": [], 'neg': []})
    for source_artifact_ids, target_artifact_ids in dataset.get_layer_iterator():
        for s_id in source_artifact_ids:
            for t_id in target_artifact_ids:
                trace_id = t_id_creator(source=s_id, target=t_id)

                if trace_id not in dataset.trace_map:
                    lookup[s_id]['neg'].append(t_id)
                else:
                    lookup[s_id]['pos'].append(t_id)

    return lookup
