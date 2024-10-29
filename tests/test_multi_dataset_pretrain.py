import os

from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.losses import ContrastiveTensionLoss

from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import create_retrieval_queries, eval_model
from trace_model_trainer.evaluation_context import EvaluationContext
from trace_model_trainer.formatters.formatter_factory import FormatterFactory
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
MODEL_NAME = "all-MiniLM-L6-v2"
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 5e-6

splitter = SplitterFactory.QUERY.create(group_col="target")


def main():
    # Create Context
    test_output_path = os.path.expanduser("~/projects/trace-model-trainer/output/multi_dataset_pretrain")
    context = EvaluationContext(test_output_path)

    # Create Datasets
    train_dataset_map, test_dataset_map = create_datasets(DATASETS)
    samples = [s for d in test_dataset_map.values() for s in create_retrieval_queries(d)]

    # Load Model
    st_model = STModel(MODEL_NAME, formatter=FormatterFactory.CONTRASTIVE_TENSION.create())

    # Create Loss
    dataset_keys = list(set(train_dataset_map.keys()).union(test_dataset_map.keys()))
    loss = ContrastiveTensionLoss(st_model.get_model())
    losses = {d_name: loss for d_name in dataset_keys}

    predictions, before_metrics = eval_model(st_model, test_dataset_map)

    # Create Trainer
    st_model.train(
        train_dataset_map,
        losses,
        output_path=os.path.join(context.get_base_path(), "model"),
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        args={
            "num_train_epochs": EPOCHS,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": BATCH_SIZE,
            "enable_full_determinism": True,
            "seed": 42,
            "eval_strategy": "epoch",
            "eval_steps": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "evaluator_map"
        },
        evaluator=RerankingEvaluator(
            samples=samples,
            batch_size=8,
            show_progress_bar=False,
            write_csv=True,
            name="evaluator"
        )
    )

    predictions, after_metrics = eval_model(st_model, test_dataset_map)
    print("BEFORE\n", before_metrics)
    print("AFTER\n", after_metrics)


def create_datasets(dataset_names):
    train_dataset_map = {}
    test_dataset_map = {}

    for dataset_name in dataset_names:
        dataset = load_traceability_dataset(f"thearod5/{dataset_name}")
        train_x, test_x = splitter.split(dataset, 0.1)

        train_dataset_map[f"{dataset_name}-train"] = train_x
        test_dataset_map[f"{dataset_name}-train"] = test_x

    return train_dataset_map, test_dataset_map


if __name__ == "__main__":
    main()
