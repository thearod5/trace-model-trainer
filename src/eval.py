from sentence_transformers import SentenceTransformer

from experiment.runner import create_experiment_dataset
from infra.eval import eval_model, eval_vsm, print_metrics
from tdata.reader import read_project


def run_eval(eval_project_path: str, model_name: str):
    test_dataset = read_project(eval_project_path)
    test_dataset_transformed = create_experiment_dataset(test_dataset)

    # Baseline scores
    m1, _ = eval_vsm(test_dataset)

    # Load SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Evaluate
    m2, _ = eval_model(model, test_dataset)

    metrics = [m1, m2]
    metric_names = ["vsm", model_name]
    print_metrics(metrics, metric_names)
