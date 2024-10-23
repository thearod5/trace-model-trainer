import os
from typing import Dict, List

from pandas import DataFrame

from trace_model_trainer.tdata.exporter import TraceDatasetExporter
from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.utils import write_json


class EvaluationContext:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metrics: List[Dict] = []
        self.run_name = None
        self.n_runs = 0

    def set_base_path(self, base_path: str = None) -> None:
        """
        Sets the current path (within output path) to start logging to.
        :param base_path: Path within output path to point current logging directory.
        :return: None
        """
        base_path = base_path or f"run{self.n_runs + 1}"
        self.run_name = None
        run_path = self.get_relative_path(base_path)
        self.run_name = base_path
        os.makedirs(run_path, exist_ok=True)
        self.n_runs += 1

    def log_dataset(self, dataset: TraceDataset, dir_name: str) -> None:
        TraceDatasetExporter.export(dataset, self.get_relative_path(dir_name))

    def log_metrics(self, metrics: Dict, **kwargs) -> Dict:
        entry = {**metrics, **kwargs}
        self.metrics.append(entry)
        return entry

    def save_json(self, content: Dict, file_name: str, pretty: bool = False) -> None:
        write_json(content, self.get_relative_path(file_name), pretty=pretty)

    def get_metrics(self) -> DataFrame:
        metric_df = DataFrame(self.metrics)
        return metric_df

    def log_df(self, df, rel_path: str):
        df.to_csv(self.get_relative_path(rel_path), index=False)

    def clear_run(self):
        self.run_name = None

    def get_relative_path(self, *sub_paths) -> str:
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        base_path = self.output_path
        if self.run_name:
            base_path = os.path.join(self.output_path, self.run_name)
        return os.path.join(base_path, *sub_paths)
