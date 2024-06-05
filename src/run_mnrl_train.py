import os.path

from sentence_transformers import SentenceTransformer

from eval import eval_model
from tdata.reader import read_project
from train_mnrl import train_mnrl

if __name__ == "__main__":
    project_path = "../res/safa"
    OUTPUT_PATH = os.path.expanduser("~/desktop/safa/output/trace-model-trainer")

    trace_dataset = read_project(project_path)

    train_dataset, test_dataset = trace_dataset.split(0.1)

    # Load SentenceTransformer model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Baseline metrics on test set
    original_metrics = eval_model(model, test_dataset, title="Pre-Training Test Metrics")

    # Train
    model = train_mnrl(train_dataset, 5, model_name, output_path=OUTPUT_PATH)

    # Evaluate after training
    eval_model(model, test_dataset, title="Post-Training Test Metrics")
    print("Original Metrics")
    print(original_metrics)
