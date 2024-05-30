import os.path

from sentence_transformers import SentenceTransformer

from eval import eval_model
from reader import read_project
from train import train_triplet
from training_data import create_training_data

if __name__ == "__main__":
    project_path = "../res/safa"
    OUTPUT_PATH = os.path.expanduser("~/desktop/safa/output/trace-model-trainer")

    project_data = read_project(project_path)
    training_data = create_training_data(project_data)

    artifact_map = project_data.artifact_map

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Baseline metrics on test set
    original_metrics = eval_model(model, training_data.test_dataset, training_data.test_dataset, title="Test Metrics")

    # Train
    model = train_triplet(training_data, 5, "../export/test-model", output_path=OUTPUT_PATH)

    # Evaluate after training
    eval_model(model, training_data.test_dataset, training_data.test_dataset, title="Test Metrics")
    print("Original Metrics")
    print(original_metrics)
