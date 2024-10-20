import os.path

from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.losses import ContrastiveLoss

from constants import BATCH_SIZE
from eval.kfold import kfold
from eval.splitters.splitter_factory import SplitterFactory
from eval.utils import create_samples, eval_model
from formatters.formatter_factory import FormatterFactory
from models.st_model import STModel
from readers.loader import load_traceability_dataset


def main():
    test_output_path = os.path.expanduser("~/desktop/safa/output/trace-model-trainer/st_test_output")
    dataset = load_traceability_dataset("thearod5/ebt")
    splitter = SplitterFactory.QUERY.create(group_col="target")

    st_model = STModel("all-MiniLM-L6-v2")
    st_formatter = FormatterFactory.CLASSIFICATION.create()
    for train_dataset, val_dataset, test_dataset in kfold(dataset, [0.60, 0.20, 0.20], splitter, 1):
        print("Before:", eval_model(st_model, test_dataset))
        evaluator = RerankingEvaluator(
            samples=create_samples(val_dataset),
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            write_csv=True,
            name="evaluator"
        )
        print("Before:", evaluator(st_model.get_model()))
        losses = ContrastiveLoss(st_model.get_model())
        train_dataset_map = {"train_dataset": st_formatter.format(train_dataset)}

        st_model.train(train_dataset_map,
                       losses,
                       output_path=test_output_path,
                       evaluator=evaluator,
                       best_metric="evaluator_map")
        print("Done training")
        print("After:", eval_model(st_model, test_dataset))


if __name__ == '__main__':
    main()
