import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from experiment.vsm import VSMController
from infra.eval import predict_model
from selection.top_candidate import select_top_candidates
from tdata.trace_dataset import TraceDataset
from tdata.utils import get_content


def create_vsm_important_words(dataset: TraceDataset, min_words: int, threshold: float, self_links: bool = True):
    def get_id(a_id):
        return f"{str(a_id)}_transformed"

    artifact_df = dataset.artifact_df.copy()
    vsm_controller = VSMController()
    vsm_controller.train(dataset.artifact_map.values())

    new_artifacts = []
    for i, a_row in artifact_df.iterrows():
        target_content = vsm_controller.get_top_n_words(get_content(a_row.to_dict()), min_words, threshold)
        new_artifacts.append({
            "id": get_id(a_row["id"]),
            "content": target_content,
            "layer": a_row["layer"]
        })

    if self_links:
        source_artifacts = list(artifact_df["id"])
        target_artifacts = [a_row['id'] for a_row in new_artifacts]
        trace_df = DataFrame({"source": source_artifacts, "target": target_artifacts, "label": [1] * len(source_artifacts)})
    else:
        trace_df = dataset.trace_df

    if self_links:
        layers = list(artifact_df["layer"].unique())
        layer_df = pd.DataFrame({"source": layers, "target": layers})
        layer_df['label'] = 1
    else:
        layer_df = dataset.layer_df
    artifact_df = pd.concat([artifact_df, DataFrame(new_artifacts)])
    return TraceDataset(artifact_df, trace_df, layer_df)


def create_bootstrapped_links(trace_dataset: TraceDataset, model: SentenceTransformer):
    _, predictions = predict_model(model, trace_dataset)
    top_candidates = select_top_candidates(predictions)
    sources = [p.source for p in top_candidates]
    targets = [p.target for p in top_candidates]
    labels = [1] * len(top_candidates)
    trace_df = DataFrame({"source": sources, "target": targets, "label": labels})
    return TraceDataset(trace_dataset.artifact_df, trace_df, trace_dataset.layer_df)
