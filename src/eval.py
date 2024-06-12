from sentence_transformers import SentenceTransformer

from infra.eval import eval_vsm, predict_model, print_metrics
from tdata.reader import read_project


def run_eval(eval_project_path: str, model_name: str):
    test_dataset = read_project(eval_project_path)
    use_vsm = False

    metrics = []
    metric_names = []

    # Baseline scores
    if use_vsm:
        m1, _ = eval_vsm(test_dataset)
        metrics.append(m1)
        metric_names.append("vsm")

    # Load SentenceTransformer model
    if not use_vsm:
        model = SentenceTransformer(model_name)
        m2, _ = predict_model(model, test_dataset)
        metrics.append(m2)
        metric_names.append(model_name)

    # Print metrics
    print_metrics(metrics, metric_names)
