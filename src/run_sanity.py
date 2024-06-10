from sentence_transformers import SentenceTransformer

from experiment.runner import create_experiment_dataset
from infra.eval import eval_model, print_metrics
from infra.generic_trainer import generic_train
from tdata.reader import read_project

if __name__ == "__main__":
    eval_project_path = "../res/safa_nl"
    model_name = "all-MiniLM-L6-v2"

    test_dataset_transformed = create_experiment_dataset(read_project(eval_project_path))

    # Load SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Evaluate
    # m2, _ = eval_model(model, test_dataset, title="Test Metrics (original)")
    m3, _ = eval_model(model, test_dataset_transformed, title="Test Metrics (transformed)")

    # Training
    trained_model = generic_train(test_dataset_transformed,
                                  "mnrl_symetric",
                                  model_name=model_name,
                                  n_epochs=3)

    m4, _ = eval_model(trained_model, test_dataset_transformed, title="Test Metrics (original-trained)")
    # m5, _ = eval_model(model, test_dataset_transformed, title="Test Metrics (transformed-trained)")

    print_metrics([m3, m4], ["original", "trained"])
