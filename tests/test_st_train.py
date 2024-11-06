import os.path

from sentence_transformers.losses import ContrastiveLoss

from trace_model_trainer.eval.kfold import kfold
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.evaluation_context import EvaluationContext
from trace_model_trainer.formatters.classification_formatter import ClassificationFormatter
from trace_model_trainer.models.st.balanced_trainer import BalancedTrainer
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.utils import clear_memory


def main():
    # os.path.expanduser("~/projects/trace-model-trainer/res/test")
    dataset = load_traceability_dataset("thearod5/cchit")
    splitter = SplitterFactory.QUERY.create(group_col="target")
    test_output_path = os.path.expanduser("~/projects/trace-model-trainer/output/st_test_output")
    context = EvaluationContext(test_output_path)

    for train_dataset, test_dataset, seed in kfold(dataset, [.8, 0.1, 0.1], splitter, 1, [42]):
        context.set_base_path(f"seed={seed}")
        st_model = STModel("all-MiniLM-L6-v2")
        _, before_metrics = eval_model(st_model, test_dataset)

        loss = ContrastiveLoss(st_model.get_model())

        trainer = st_model.train(
            train_dataset={"train_data": ClassificationFormatter().format(train_dataset)},
            eval_dataset={"eval_data": ClassificationFormatter().format(test_dataset)},
            losses={"train_data": loss, "eval_data": loss},
            output_path=context.get_relative_path("model"),
            trainer_class=BalancedTrainer,
            batch_size=8,
            learning_rate=5e-6,
            args={
                "num_train_epochs": 2,
                "enable_full_determinism": True,
                "seed": seed,
                "eval_strategy": "epoch",
                "eval_steps": 1,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_eval_data_loss",
                "logging_dir": context.get_relative_path("output"),
            }
        )
        _, after_metrics = eval_model(st_model, test_dataset)
        print(before_metrics)
        print(after_metrics)

        for log in trainer.state.log_history:
            print(log)

        del st_model
        st_model = None
        clear_memory()

    metric_df = context.get_metrics()
    context.log_df(metric_df, "metrics.csv")


if __name__ == '__main__':
    main()
