import os.path

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss

from st3traceability.constants import TARGET_COL
from st3traceability.formatters.formatter_factory import FormatterFactory
from st3traceability.kfold import kfold
from st3traceability.loader import load_traceability_dataset
from st3traceability.splitters.splitter_factory import SplitterFactory
from st3traceability.training.training_context import TrainingContext


def train_task(dataset: Dataset, model_name: str):
    formatter = FormatterFactory.ANCHOR_POSITIVE.create()
    splitter = SplitterFactory.QUERY.create(group_col=TARGET_COL)

    for train_dataset, val_dataset, eval_dataset in kfold(dataset, (.8, .1, .1), splitter, 5):
        output_path = os.path.expanduser("~/desktop/safa/output/test1")
        train_dataset = formatter.format(train_dataset)
        eval_dataset = formatter.format(eval_dataset)
        model: SentenceTransformer = SentenceTransformer(model_name)
        loss_func = MultipleNegativesRankingLoss(model)
        train_dataset_map = {"ebt-train": train_dataset}
        eval_dataset_map = {"ebt-eval": eval_dataset}
        losses = {"ebt-train": loss_func, "ebt-eval": loss_func}
        trainer = SentenceTransformerTrainer(
            model=model,
            args=TrainingContext.create_args(output_path),
            train_dataset=train_dataset_map,
            eval_dataset=eval_dataset_map,
            loss=losses
        )
        trainer.train()


if __name__ == "__main__":
    train_task(load_traceability_dataset("thearod5/cm1"), "sentence-transformers/all-MiniLM-L6-v2")
