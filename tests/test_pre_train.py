import os.path

from sentence_transformers.losses import ContrastiveTensionLoss

from trace_model_trainer.eval.kfold import kfold
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.evaluation_context import EvaluationContext
from trace_model_trainer.formatters.formatter_factory import FormatterFactory
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.utils import clear_memory


def main():
    test_output_path = os.path.expanduser("~/projects/trace-model-trainer/output/st_test_output")
    dataset = load_traceability_dataset("thearod5/EasyClinic")
    splitter = SplitterFactory.QUERY.create(group_col="target")
    context = EvaluationContext(test_output_path)

    context.log_dataset(dataset, "dataset")

    st_model = STModel("all-MiniLM-L6-v2", formatter=FormatterFactory.CONTRASTIVE_TENSION.create())
    for train_dataset, val_dataset, test_dataset, seed in kfold(dataset, [0.80, 0.1, 0.1], splitter, 1, [42]):
        context.set_base_path(f"seed={seed}")
        context.log_dataset(test_dataset, "test")
        context.log_dataset(train_dataset, "train")

        _, metrics = eval_model(st_model, test_dataset)
        print("BEFORE\n", metrics)
        loss = ContrastiveTensionLoss(st_model.get_model())
        st_model.train(dataset,
                       loss,
                       eval_dataset=val_dataset,
                       output_path=context.get_relative_path("model"),
                       args={"num_train_epochs": 2, "enable_full_determinism": True, "seed": seed}
                       )
        predictions, metrics = eval_model(st_model, test_dataset)
        print("AFTER\n", metrics)

        clear_memory()

    metric_df = context.get_metrics()
    context.log_df(metric_df, "metrics.csv")


if __name__ == '__main__':
    main()
