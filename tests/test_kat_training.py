import os

from sentence_transformers.losses import ContrastiveTensionLoss

from trace_model_trainer.eval.splitters.query_splitter import QuerySplitter
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.evaluation_context import EvaluationContext
from trace_model_trainer.formatters.contrastive_tension_formatter import ContrastiveTensionFormatter
from trace_model_trainer.formatters.kat_formatter import KatFormatter
from trace_model_trainer.models.st.balanced_trainer import BalancedTrainer
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.tdata.loader import load_traceability_dataset

DATASETS = [
    "eTour",  # Neutral
    "CCHIT",
    "eAnci",  # Good
    "EasyClinic",
    "iTrust",  # Good
    "IceBreaker",  # Neutral
    "InfusionPump",
    "SMOS",  # Good
    "Albergate",
    "GANNT"
]
DATASETS = DATASETS[:1]
MODEL_NAME = "all-MiniLM-L6-v2"
EPOCHS = 1
BATCH_SIZE = 3
LEARNING_RATE = 5e-6

splitter = SplitterFactory.QUERY.create(group_col="target")


def main():
    # Create Context
    test_output_path = os.path.expanduser("~/projects/trace-model-trainer/output/multi_dataset_pretrain")
    context = EvaluationContext(test_output_path)

    # Create Datasets
    #
    dataset = load_traceability_dataset(os.path.expanduser("~/projects/trace-model-trainer/res/test"))
    train_data, test_data = QuerySplitter().split(dataset, .2, random_seed=42)

    # Load Model
    st_model = STModel(MODEL_NAME)

    # Baseline Evaluation
    _, before_metrics = eval_model(st_model, test_data)

    # Create Loss
    loss = ContrastiveTensionLoss(st_model.get_model())

    # Create Trainer
    st_model.train(
        train_dataset={"train": KatFormatter().format(train_data, dataset.artifact_df)},
        eval_dataset={"test_data": ContrastiveTensionFormatter().format(test_data)},
        losses={"train": loss, "test_data": loss},
        output_path=os.path.join(context.get_base_path(), "model"),
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        trainer_class=BalancedTrainer,
        args={
            "num_train_epochs": EPOCHS,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": BATCH_SIZE,
            "enable_full_determinism": True,
            "seed": 42,
            "eval_strategy": "epoch",
            "eval_steps": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_test_data_loss"
        }
    )

    _, after_metrics = eval_model(st_model, test_data)
    print("BEFORE\n", before_metrics["dataset"])
    print("AFTER\n", after_metrics["dataset"])


if __name__ == "__main__":
    main()
