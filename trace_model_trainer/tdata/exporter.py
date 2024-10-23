import os.path

from trace_model_trainer.tdata.trace_dataset import TraceDataset


class TraceDatasetExporter:
    @staticmethod
    def export(dataset: TraceDataset, export_dir: str):
        os.makedirs(export_dir, exist_ok=True)
        dataset.artifact_df.to_csv(os.path.join(export_dir, "artifacts.csv"), index=False)
        dataset.trace_df.to_csv(os.path.join(export_dir, "traces.csv"), index=False)
        dataset.layer_df.to_csv(os.path.join(export_dir, "matrices.csv"), index=False)
