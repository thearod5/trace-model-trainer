#
# Install Necessary Packages
#
import os

from trace_model_trainer.tdata.loader import load_traceability_dataset

RUN_PATH = os.path.join(os.path.expanduser("~/projects/trace-model-trainer/output/test"), "rq1", "data", "artifact_training")

#
# REPO IMPORTS
#
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.utils import clear_memory
from trace_model_trainer.formatters.contrastive_tension_formatter import ContrastiveTensionFormatter
from trace_model_trainer.models.st.balanced_trainer import BalancedTrainer

from sentence_transformers.losses import ContrastiveTensionLoss

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DATASETS = [
#     "eTour", # Neutral
#     "CCHIT",
#     "eAnci", # Good
#     "EasyClinic",
#     "iTrust",  # Good
#     "IceBreaker",# Neutral
#     "InfusionPump",
#     "SMOS", # Good
#     "Albergate",
#     "GANNT"
#     ]
dataset_name = "eAnci"
MODEL_NAME = "all-MiniLM-L6-v2"
SPLITS = [0.1, 0.1, 0.8]
EPOCHS = 5
N_FOLDS = 5
BATCH_SIZE = 128
LEARNING_RATE = 5e-6
RANDOM_SEEDS = [364882, 332305, 351071, 804707, 191934]


def main():
    #
    # START
    #
    splitter = SplitterFactory.QUERY.create(group_col="target")
    print("Dataset:", dataset_name)
    # Load Dataset
    dataset = load_traceability_dataset(f"thearod5/{dataset_name}")
    run_path = os.path.join(RUN_PATH, dataset_name)
    # assert not os.path.exists(run_path), f"{run_path} already exists."

    metrics = []

    st_model = STModel(MODEL_NAME)

    # Baseline Evaluation
    _, baseline_metrics = eval_model(st_model, dataset)
    metrics.append({
        "condition": "baseline",
        **baseline_metrics["dataset"]
    })

    # Artifact Training
    loss_func = ContrastiveTensionLoss(st_model.get_model())

    trainer = st_model.train(
        train_dataset={
            "train_data": ContrastiveTensionFormatter().format(dataset)
        },
        eval_dataset={
            "val": ContrastiveTensionFormatter().format(dataset)
        },
        losses={
            "train_data": loss_func,
            "val": loss_func
        },

        output_path=os.path.join(run_path, "artifact_model"),
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        trainer_class=BalancedTrainer,
        args={
            "num_train_epochs": EPOCHS,
            "enable_full_determinism": True,
            "seed": 42,
            "eval_strategy": "epoch",
            "eval_steps": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_val_loss"
        }
    )

    # Artifact Training Evaluation
    _, trained_metrics = eval_model(st_model, dataset)
    metrics.append({
        "condition": "artifact_trained",
        **trained_metrics["dataset"]
    })
    print(trained_metrics["dataset"])

    # Cleanup
    del st_model
    st_model = None
    clear_memory()


if __name__ == "__main__":
    main()
