import os

from sentence_transformers.losses import AnglELoss, ContrastiveTensionLoss

from trace_model_trainer.eval.splitters.query_splitter import QuerySplitter
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.evaluation_context import EvaluationContext
from trace_model_trainer.formatters.artifact_augmentation_formatter import ArtifactAugmentationFormatter
from trace_model_trainer.formatters.contrastive_tension_formatter import ContrastiveTensionFormatter
from trace_model_trainer.loss.custom_cosine_loss import CustomCosineEvaluator
from trace_model_trainer.models.st.balanced_trainer import AugmentedTrainer, BalancedTrainer
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.tdata.loader import load_traceability_dataset

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 4
LEARNING_RATE = 5e-6
EPOCHS = 2


def main():
    # Create Context
    test_output_path = os.path.expanduser("~/projects/trace-model-trainer/output/multi_dataset_pretrain")
    context = EvaluationContext(test_output_path)

    # Create Datasets
    # os.path.expanduser("~/projects/trace-model-trainer/res/test")
    # 364882
    dataset = load_traceability_dataset(os.path.expanduser("~/projects/trace-model-trainer/res/test"))
    splitter = QuerySplitter()
    train_dataset, test_dataset = splitter.split(dataset, 2 / 3)
    train_dataset, val_dataset = splitter.split(train_dataset, 1 / 2)

    # Load Model
    st_model = STModel(MODEL_NAME)

    # Baseline Evaluation
    _, metrics0 = eval_model(st_model, dataset)
    print("Metrics (0):", metrics0["dataset"])

    # Create Loss
    artifact_loss = ContrastiveTensionLoss(st_model.get_model())

    # Create Trainer
    trainer = st_model.train(
        train_dataset={"train": ContrastiveTensionFormatter().format(dataset)},
        losses={"train": artifact_loss},
        output_path=os.path.join(context.get_base_path(), "model"),
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        trainer_class=BalancedTrainer,
        evaluator=CustomCosineEvaluator(val_dataset),
        args={
            "num_train_epochs": EPOCHS,
            "enable_full_determinism": True,
            "seed": 42,
            "eval_strategy": "epoch",
            "eval_steps": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_map",
            "greater_is_better": True
        },
    )

    _, metrics1 = eval_model(st_model, dataset)
    print("Metrics 0:\n", metrics0["dataset"])
    print("Metrics 1:\n", metrics1["dataset"])

    trace_loss = AnglELoss(st_model.get_model())
    st_model.train(
        train_dataset={"train_data": ArtifactAugmentationFormatter().format(train_dataset)},
        losses={"train_data": trace_loss},
        output_path=context.get_relative_path("model"),
        trainer_class=AugmentedTrainer,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluator=CustomCosineEvaluator(val_dataset),
        args={
            "num_train_epochs": 2,
            "enable_full_determinism": True,
            "seed": 42,
            "eval_strategy": "epoch",
            "eval_steps": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_map",
            "greater_is_better": True
        }
    )

    _, metrics2 = eval_model(st_model, dataset)
    print("Metrics 0:\n", metrics0["dataset"])
    print("Metrics 1:\n", metrics1["dataset"])
    print("Metrics 2:\n", metrics2["dataset"])


if __name__ == "__main__":
    main()
