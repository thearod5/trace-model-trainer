from sentence_transformers import SentenceTransformer

from experiment.runner import create_experiment_dataset
from infra.eval import eval_model, print_metrics
from infra.generic_trainer import generic_train
from tdata.reader import read_project
from utils import get_or_prompt

if __name__ == "__main__":
    eval_project_path = get_or_prompt("EVAL_PROJECT_PATH", "Eval Project Path: ")
    model_name = get_or_prompt("MODEL", "Model: ")

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
                  ["original (baseline)", "original (trained)"])
    print_metrics([m1, m4],
                  ["original (baseline)", "transformed (baseline)", "original (trained)", "transformed (trained)"])
