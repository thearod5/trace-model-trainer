from sentence_transformers import SentenceTransformer

from eval import eval_model
from reader import read_project
from train import train_triplet
from training_data import create_training_data

if __name__ == "__main__":
    project_path = "../res/safa"

    project_data = read_project(project_path)
    training_data = create_training_data(project_data)

    artifact_map = project_data.artifact_map

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Baseline metrics on test set
    eval_model(model, training_data.test_dataset, training_data.test_dataset, title="Test Metrics")

    # Train
    train_triplet(training_data, "../export/test-model.pth")

    # Evaluate after training
    eval_model(model, training_data.test_dataset, training_data.test_dataset, title="Test Metrics")
