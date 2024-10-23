import os
from typing import Dict, List

from pandas import DataFrame

from trace_model_trainer.tdata.exporter import TraceDatasetExporter
from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.utils import write_json


class EvaluationContext:
    def __init__(self, output_path: str, dataset_dir_name: str = "dataset",
                 metric_name: str = "metrics.json"):
        assert metric_name.endswith(".json"), "metric_name must end with .json"
        self.output_path = output_path
        self.metrics: List[Dict] = []
        self.dataset_dir_name = dataset_dir_name
        self.metric_name = metric_name
        self.run_name = None
        self.n_runs = 0

    def init_run(self, run_name: str = None):
        self.run_name = run_name or f"run{self.n_runs + 1}"
        self.run_name = None
        run_path = self.get_relative_path(run_name)
        self.run_name = run_name
        os.makedirs(run_path, exist_ok=True)

    def log_dataset(self, dataset: TraceDataset) -> None:
        TraceDatasetExporter.export(dataset, self.get_relative_path(self.dataset_dir_name))

    def log_metrics(self, **kwargs) -> None:
        if self.run_name:
            kwargs["run"] = self.run_name
        self.metrics.append(kwargs)

    def save_json(self, content: Dict, file_name: str, pretty: bool = False) -> None:
        write_json(content, self.get_relative_path(file_name), pretty=pretty)

    def done(self) -> None:
        self.run_name = None
        metric_df = DataFrame(self.metrics)
        metric_df.to_csv(self.get_relative_path(self.metric_name), index=False)

    def get_relative_path(self, *sub_paths) -> str:
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        base_path = self.output_path
        if self.run_name:
            base_path = os.path.join(self.output_path, self.run_name)
        return os.path.join(base_path, *sub_paths)
