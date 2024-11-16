import os

from datasets import Dataset
from sklearn.model_selection import train_test_split

from scripts.find_important_words import get_nouns
from scripts.token_importance import calculate_token_importance
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.loss.custom_cosine_loss import CustomCosineEvaluator
from trace_model_trainer.loss.noun_loss import NounLoss
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.tdata.loader import load_traceability_dataset


def runner():
    output_path = os.path.expanduser("~/projects/trace-model-trainer/output/nouns")

    p0 = os.path.expanduser("~/projects/trace-model-trainer/res/test")
    p1 = "thearod5/ebt"
    dataset = load_traceability_dataset(p0)
    texts = list(dataset.artifact_map.values())
    train_texts, val_texts = train_test_split(texts, train_size=0.9)

    # Load the SentenceTransformer model
    st_model = STModel("all-MiniLM-L6-v2")

    # Evaluation
    calculate_token_importance(st_model.get_model(), val_texts, os.path.join(output_path, "before.png"))
    _, before_metrics = eval_model(st_model, dataset)
    print("Before metrics: ", before_metrics['dataset'])

    # Loss Setup
    important_words, text2words = get_nouns(texts)
    loss = NounLoss(st_model.get_model(), important_words)

    # Training

    trainer = st_model.train(
        train_dataset={
            "train_data": Dataset.from_dict({"sentence1": train_texts})
        },
        losses={"train_data": loss, "eval_data": loss},
        output_path=output_path,
        batch_size=4,
        learning_rate=5e-5,
        evaluator=CustomCosineEvaluator(dataset),
        args={
            "num_train_epochs": 5,
            "enable_full_determinism": True,
            "seed": 42,
            "eval_strategy": "epoch",
            "eval_steps": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_map",
            "greater_is_better": True,
        }
    )

    # Post-Training Evaluation
    calculate_token_importance(st_model.get_model(), val_texts, os.path.join(output_path, "after.png"))
    _, after_metrics = eval_model(st_model, dataset)
    print(before_metrics)
    print(after_metrics)


if __name__ == "__main__":
    runner()
