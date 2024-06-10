from sentence_transformers import SentenceTransformer

from experiment.vsm import VSMController
from infra.eval import eval_model, print_metrics
from infra.generic_trainer import generic_train
from tdata.reader import read_project
from tdata.trace_dataset import TraceDataset


def create_experiment_dataset(dataset: TraceDataset, min_words: int = 5):
    artifact_df = dataset.artifact_df.copy()

    vsm_controller = VSMController()
    vsm_controller.train(dataset.artifact_map.values())

    artifact_df["content"] = artifact_df["content"].apply(lambda a_content: vsm_controller.get_top_n_words(a_content, min_words))

    return TraceDataset(artifact_df, dataset.trace_df, dataset.layer_df)


def run_experiment(eval_project_path: str, model_name: str):
    test_dataset = read_project(eval_project_path)
    test_dataset_transformed = create_experiment_dataset(read_project(eval_project_path))
    # Load SentenceTransformer model
    model = SentenceTransformer(model_name)
    # Evaluate
    m1, _ = eval_model(model, test_dataset)
    m2, _ = eval_model(model, test_dataset_transformed)
    # Training
    trained_model = generic_train(test_dataset_transformed,
                                  "mnrl_symetric",
                                  model_name=model_name,
                                  n_epochs=20,
                                  disable_tqdm=True)
    m3, _ = eval_model(trained_model, test_dataset)
    m4, _ = eval_model(trained_model, test_dataset_transformed)
    print_metrics([m1, m3],
                  ["eval data (baseline)", "eval data (post-training)"])
    print_metrics([m1, m4],
                  ["training data (baseline)", "training data (trained)"])
