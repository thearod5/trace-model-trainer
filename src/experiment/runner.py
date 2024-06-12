import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from experiment.vsm import VSMController
from infra.eval import eval_model, print_metrics
from infra.generic_trainer import generic_train
from tdata.reader import read_project
from tdata.trace_dataset import TraceDataset
from tdata.utils import get_content
from utils import clear_memory


def create_experiment_dataset(dataset: TraceDataset, min_words: int, threshold: float):
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


def run_experiment(eval_project_path: str, model_name: str, min_words: 5, threshold=3.0):
    test_dataset = read_project(eval_project_path)
    test_dataset_transformed = create_experiment_dataset(read_project(eval_project_path),
                                                         min_words=min_words,
                                                         threshold=threshold)

    # Load SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Before Training Evaluate
    m1, _ = eval_model(model, test_dataset)
    clear_memory(model)

    # Training
    trained_model = generic_train(test_dataset_transformed,
                                  "mnrl_symetric",
                                  model_name=model_name,
                                  n_epochs=20,
                                  disable_tqdm=True)
    # Eval
    m3, _ = eval_model(trained_model, test_dataset)
    print_metrics([m1, m3],
                  ["before-training", "after-training"])
