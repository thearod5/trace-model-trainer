from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from experiment.runner import create_experiment_dataset
from infra.eval import eval_model, eval_vsm, print_metrics
from tdata.reader import read_project
from utils import get_or_prompt

if __name__ == "__main__":
    load_dotenv()
    eval_project_path = get_or_prompt("EVAL_PROJECT_PATH", "Eval Project Path: ")
    model_name = get_or_prompt("MODEL", "Model: ")

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
