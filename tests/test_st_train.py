import os.path

from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.losses import ContrastiveLoss

from trace_model_trainer.eval.kfold import kfold
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import create_retrieval_queries, eval_model
from trace_model_trainer.evaluation_context import EvaluationContext
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.utils import clear_memory


def main():
    dataset = load_traceability_dataset("thearod5/EasyClinic")
    splitter = SplitterFactory.QUERY.create(group_col="target")
    test_output_path = os.path.expanduser("~/projects/trace-model-trainer/output/st_test_output")
    context = EvaluationContext(test_output_path)

    st_model = STModel("all-MiniLM-L6-v2")
    for train_dataset, val_dataset, test_dataset, seed in kfold(dataset, [0.1, 0.3, 0.6], splitter, 1, [42]):
        context.set_base_path(f"seed={seed}")

        _, before_metrics = eval_model(st_model, test_dataset)
        loss = ContrastiveLoss(st_model.get_model())
        losses = {"train-test": loss, "eval-test": loss}
        trainer = st_model.train({"train-test": train_dataset},
                                 loss,
                                 output_path=context.get_relative_path("model"),
                                 eval_dataset={"eval-test": val_dataset},
                                 args={
                                     "num_train_epochs": 2,
                                     "enable_full_determinism": True,
                                     "seed": seed,
                                     "eval_strategy": "epoch",
                                     "eval_steps": 1,
                                     "load_best_model_at_end": True,
                                     "metric_for_best_model": "evaluator_map",
                                     "logging_dir": context.get_relative_path("output"),
                                 },
                                 evaluator=RerankingEvaluator(
                                     samples=create_retrieval_queries(val_dataset),
                                     batch_size=8,
                                     show_progress_bar=False,
                                     write_csv=True,
                                     name="evaluator"),
                                 compute_metrics=compute_metrics_handler)
        _, after_metrics = eval_model(st_model, test_dataset)
        print(before_metrics)
        print(after_metrics)

        for log in trainer.state.log_history:
            print(log)

        clear_memory()

    metric_df = context.get_metrics()
    context.log_df(metric_df, "metrics.csv")


def compute_metrics_handler(*args, **kwargs):
    print("computing...")


if __name__ == '__main__':
    main()
