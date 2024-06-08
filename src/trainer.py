from typing import List

from sentence_transformers import SentenceTransformer

from constants import OUTPUT_PATH
from infra.eval import eval_model, print_links
from infra.generic_trainer import generic_train
from tdata.reader import read_project
from tdata.types import TracePrediction
from utils import t_id_creator


def compare(a_preds: List[TracePrediction], b_preds: List[TracePrediction]):
    a_missed = {t_id_creator(source=t.source, target=t.target): t for t in a_preds if t.score < 0.5 and t.label == 1}
    b_missed = {t_id_creator(source=t.source, target=t.target): t for t in b_preds if t.score < 0.5 and t.label == 1}

    b_only_links = set(b_missed.keys()).difference(a_missed.keys())
    print_links([b_missed[t_id] for t_id in b_only_links])


if __name__ == "__main__":
    train_project_path = "../res/vsm_experiment"
    eval_project_path = "../res/safa"
    loss_func_name = "mnrl_symetric"
    model_name = "all-MiniLM-L6-v2"
    #
    # Training and Evaluation
    #

    train_dataset = read_project(train_project_path)
    test_dataset = read_project(eval_project_path)

    # Load SentenceTransformer model

    model = SentenceTransformer(model_name)

    # Baseline metrics on test set
    original_metrics, original_predictions = eval_model(model, test_dataset, title="Pre-Training Test Metrics")

    # Train
    trained_model = generic_train(train_dataset,
                                  loss_func_name,
                                  output_path=OUTPUT_PATH)

    # Evaluate after training
    trained_metrics, trained_predictions = eval_model(trained_model, test_dataset, title="Post-Training Test Metrics")
    print("Original Metrics")
    print(original_metrics)

    compare(original_predictions, trained_predictions)
