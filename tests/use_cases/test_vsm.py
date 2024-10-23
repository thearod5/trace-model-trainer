import os.path
import shutil

from datasets import DownloadMode

from trace_model_trainer.eval.kfold import kfold
from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.evaluation_context import EvaluationContext
from trace_model_trainer.models.vsm_model import VSMModel
from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.utils import group_by


def main():
    OUTPUT_PATH = "~/projects/trace-model-trainer/output"

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
        os.makedirs(OUTPUT_PATH)

    context = EvaluationContext(os.path.expanduser(OUTPUT_PATH))

    dataset = load_traceability_dataset("thearod5/eTour", download_mode=DownloadMode.FORCE_REDOWNLOAD)
    splitter = SplitterFactory.QUERY.create(group_col="target")

    vsm_model = VSMModel()
    i = 0
    for train_dataset, test_dataset in kfold(dataset, [0.80, 0.20], splitter, 3):
        context.init_run(f"run{i + 1}")
        context.log_dataset(test_dataset)
        context.save_json({"vsm_test": True}, "config.json")

        vsm_model.train(train_dataset)

        predictions, vsm_test_metrics = eval_model(vsm_model, test_dataset)
        context.save_json(group_by(predictions, "target"), "predictions.json")
        context.log_metrics(**vsm_test_metrics)
        i += 1

    context.done()


if __name__ == '__main__':
    main()
