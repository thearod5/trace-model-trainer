import os

from datasets import DatasetDict
from sentence_transformers.losses import AnglELoss

from trace_model_trainer.eval.splitters.query_splitter import QuerySplitter
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.trace_iterator import trace_iterator_labeled
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.evaluation_context import EvaluationContext
from trace_model_trainer.loss.custom_cosine_loss import CustomCosineEvaluator
from trace_model_trainer.models.st.balanced_trainer import AugmentedTrainer
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.tdata.trace_dataset import TraceDataset

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
EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

splitter = SplitterFactory.QUERY.create(group_col="target")


def main():
    # Create Context
    test_output_path = os.path.expanduser("~/projects/trace-model-trainer/output/multi_dataset_pretrain")
    context = EvaluationContext(test_output_path)

    # Create Datasets
    # os.path.expanduser("~/projects/trace-model-trainer/res/test")
    # 364882
    dataset = load_traceability_dataset(os.path.expanduser("~/projects/trace-model-trainer/res/test"))
    splitter = QuerySplitter()
    train_dataset, test_dataset = splitter.split(dataset, train_size=2 / 3)
    train_dataset, val_dataset = splitter.split(train_dataset, train_size=1 / 2)

    # Load Model
    st_model = STModel(MODEL_NAME)

    # Baseline Evaluation
    _, before_metrics = eval_model(st_model, dataset)
    print("Before Metrics:", before_metrics["dataset"])

    # Create Loss
    # loss = ContrastiveTensionLoss(st_model.get_model())
    loss = AnglELoss(st_model.get_model())

    # Create Trainer
    st_model.train(
        train_dataset={
            "train": DatasetDict({
                "traces": get_traces(train_dataset),
                "artifact_map": dataset.artifact_map,
            })
        },
        losses={"train": loss},
        output_path=os.path.join(context.get_base_path(), "model"),
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        trainer_class=AugmentedTrainer,
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

    _, after_metrics = eval_model(st_model, dataset)
    print("BEFORE\n", before_metrics["dataset"])
    print("AFTER\n", after_metrics["dataset"])


def get_traces(dataset: TraceDataset):
    traces = []
    for s_id, t_id, label in trace_iterator_labeled(dataset):
        traces.append({
            "source": dataset.artifact_map[s_id],
            "target": dataset.artifact_map[t_id],
            "label": label
        })
    return traces


if __name__ == "__main__":
    main()
