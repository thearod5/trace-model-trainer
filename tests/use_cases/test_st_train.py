import os.path

from sentence_transformers.losses import ContrastiveLoss

from trace_model_trainer.eval.kfold import kfold
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.evaluation_context import EvaluationContext
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.utils import clear_memory


def main():
    test_output_path = os.path.expanduser("~/projects/trace-model-trainer/output/st_test_output")
    dataset = load_traceability_dataset("thearod5/GANNT")
    splitter = SplitterFactory.QUERY.create(group_col="target")
    context = EvaluationContext(test_output_path)

    context.log_dataset(dataset, "dataset")

    st_model = STModel("all-MiniLM-L6-v2")
    for train_dataset, val_dataset, test_dataset in kfold(dataset, [0.60, 0.20, 0.20], splitter, 1, [42]):
        context.init_run()

        predictions, metrics = eval_model(st_model, test_dataset)
        context.log_metrics(metrics, trial="before")
        context.log_dataset(test_dataset, "test")

        loss = ContrastiveLoss(st_model.get_model())
        st_model.train(train_dataset,
                       loss,
                       output_path=test_output_path,
                       eval_dataset=val_dataset,
                       args={"num_train_epochs": 1}
                       )
        predictions, metrics = eval_model(st_model, test_dataset)
        context.log_metrics(metrics, trial="before")
        context.log_dataset(test_dataset, "test")

        clear_memory()

    context.save_metrics("metrics.csv")


if __name__ == '__main__':
    main()
