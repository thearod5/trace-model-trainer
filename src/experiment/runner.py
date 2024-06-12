import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from experiment.vsm import VSMController
from infra.eval import predict_model, print_metrics
from infra.generic_trainer import generic_train
from selection.top_candidate import select_top_candidates
from tdata.reader import read_project
from tdata.trace_dataset import TraceDataset
from tdata.utils import get_content


def create_vsm_important_words(dataset: TraceDataset, min_words: int, threshold: float):
    artifact_df = dataset.artifact_df.copy()
    vsm_controller = VSMController()
    vsm_controller.train(dataset.artifact_map.values())
    artifact_df["content"] = [vsm_controller.get_top_n_words(get_content(a_row.to_dict()), min_words, threshold)
                              for i, a_row in artifact_df.iterrows()]
    # TODO: Replace with get content
    artifacts = list(artifact_df["id"])
    trace_df = DataFrame({"source": artifacts, "target": artifacts, "label": [1] * len(artifacts)})

    layers = list(artifact_df["layer"].unique())
    layer_df = pd.DataFrame({"source": layers, "target": layers})

    return TraceDataset(artifact_df, trace_df, layer_df)


def create_bootstrapped_links(trace_dataset: TraceDataset, model: SentenceTransformer):
    _, predictions = predict_model(model, trace_dataset)
    top_candidates = select_top_candidates(predictions)
    sources = [p.source for p in top_candidates]
    targets = [p.target for p in top_candidates]
    labels = [1] * len(top_candidates)
    trace_df = DataFrame({"source": sources, "target": targets, "label": labels})
    return TraceDataset(trace_dataset.artifact_df, trace_df, trace_dataset.layer_df)


VSM_IMPORTANCE_TYPE = "vsm_importance"
BOOTSTRAP_TYPE = "bootstrap"
experiments = {VSM_IMPORTANCE_TYPE, BOOTSTRAP_TYPE}


def run_experiment(eval_project_path: str, model_name: str, experiment_type=BOOTSTRAP_TYPE):
    # Load SentenceTransformer model
    model = SentenceTransformer(model_name)
    test_dataset = read_project(eval_project_path)
    if experiment_type == VSM_IMPORTANCE_TYPE:
        test_dataset_transformed = create_vsm_important_words(read_project(eval_project_path),
                                                              min_words=5,
                                                              threshold=3.0)
    elif experiment_type == BOOTSTRAP_TYPE:
        test_dataset_transformed = create_bootstrapped_links(test_dataset, model)
    else:
        raise Exception(f"{experiment_type} is not one of {experiments}")

    # Before Training Evaluate
    m1, _ = predict_model(model, test_dataset)

    # Training
    trained_model = generic_train(test_dataset_transformed,
                                  "mnrl_symetric",
                                  model=model,
                                  n_epochs=1,
                                  disable_tqdm=True)
    # Eval
    m3, _ = predict_model(trained_model, test_dataset)
    print_metrics([m1, m3],
                  ["before-training", "after-training"])
