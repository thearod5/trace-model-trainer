import os.path

from use_cases.kfold import kfold
from use_cases.splitters.splitter_factory import SplitterFactory
from use_cases.utils import eval_model

from models.mlm_model import MLMModel
from readers.loader import load_traceability_dataset


def main():
    test_output_path = os.path.expanduser("~/desktop/safa/output/trace-model-trainer/mlm_test_output")
    dataset = load_traceability_dataset("thearod5/ebt")
    splitter = SplitterFactory.QUERY.create(group_col="target")
    mlm_metrics = []

    mlm_model = MLMModel("bert-base-uncased")
    for train_dataset, test_dataset in kfold(dataset, [0.80, 0.20], splitter, 1):
        print("Before:", eval_model(mlm_model, test_dataset))
        mlm_model.train(train_dataset, test_output_path)
        print("Done training")
        print("After:", eval_model(mlm_model, test_dataset))

    print("MLM Metrics:", mlm_metrics)


if __name__ == '__main__':
    main()
