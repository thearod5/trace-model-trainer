from trace_model_trainer.eval.kfold import kfold
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import aggregate_metrics, eval_model
from trace_model_trainer.models.vsm_model import VSMModel
from trace_model_trainer.readers.loader import load_traceability_dataset


def main():
    dataset = load_traceability_dataset("thearod5/GANNT")
    splitter = SplitterFactory.QUERY.create(group_col="target")
    vsm_metrics = []

    vsm_model = VSMModel()

    for train_dataset, test_dataset in kfold(dataset, [0.80, 0.20], splitter, 1):
        vsm_model.train(train_dataset)
        vsm_test_metrics = eval_model(vsm_model, test_dataset)
        vsm_metrics.append(vsm_test_metrics)

    print("VSM Metrics:", aggregate_metrics(vsm_metrics))


if __name__ == '__main__':
    main()
