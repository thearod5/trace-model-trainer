import os.path

from sentence_transformers import SentenceTransformer

from eval import eval_model
from tdata.reader import read_project
from train_triplet import train_triplet

if __name__ == "__main__":
    project_path = "../res/safa"
    OUTPUT_PATH = os.path.expanduser("~/desktop/safa/output/trace-model-trainer")

    trace_dataset = read_project(project_path)
    train_trace_dataset, test_trace_dataset = trace_dataset.split(0.1)

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Baseline metrics on test set
    original_metrics = eval_model(model, test_trace_dataset, title="Pre-Training Test Metrics")

    # Train
    model = train_triplet(train_trace_dataset, 5, "../export/test-model", output_path=OUTPUT_PATH)

    # Evaluate after training
    print("Original Metrics")
    print(original_metrics)
    eval_model(model, test_trace_dataset, title="Post-Training Test Metrics")
