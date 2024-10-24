from trace_model_trainer.eval.kfold import kfold
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import aggregate_metrics, eval_model
from trace_model_trainer.models.mlm_model import MLMModel
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.models.vsm_model import VSMModel
from trace_model_trainer.tdata.loader import load_traceability_dataset


def main():
    dataset = load_traceability_dataset("thearod5/ebt")
    splitter = SplitterFactory.QUERY.create(group_col="target")
    st_metrics = []
    vsm_metrics = []
    mlm_metrics = []

    vsm_model = VSMModel()
    st_model = STModel("all-MiniLM-L6-v2")
    mlm_model = MLMModel("bert-base-uncased")
    for train_dataset, test_dataset in kfold(dataset, [0.80, 0.20], splitter, 1):
        vsm_model.train(train_dataset)

        vsm_test_metrics = eval_model(vsm_model, test_dataset)
        st_test_metrics = eval_model(st_model, test_dataset)
        mlm_test_metrics = eval_model(mlm_model, test_dataset)

        vsm_metrics.append(vsm_test_metrics)
        st_metrics.append(st_test_metrics)
        mlm_metrics.append(mlm_test_metrics)

    print("ST Metrics:", aggregate_metrics(st_metrics))
    print("VSM Metrics:", aggregate_metrics(vsm_metrics))
    print("MLM Metrics:", aggregate_metrics(mlm_metrics))


if __name__ == '__main__':
    main()
