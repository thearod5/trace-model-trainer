from unittest import TestCase

from pandas import DataFrame

from trace_model_trainer.eval.splitters.query_splitter import QuerySplitter
from trace_model_trainer.tdata.trace_dataset import TraceDataset


class TestQuerySplitter(TestCase):
    def test_use_case(self):
        query_splitter = QuerySplitter()
        dataset = TraceDataset(
            artifact_df=DataFrame({
                "id": ["S1", "T1", "T2"],
                "content": ["B1", "B2", "B3"],
                "layer": ["source_type", "target_type", "target_type"]
            }),
            trace_df=DataFrame({
                "source": ["S1", "S1"],
                "target": ["T1", "T2"],
                "label": [1, 1]
            }),
            layer_df=DataFrame({
                "source_type": ["source_type"],
                "target_type": ["target_type"]
            })
        )
        train_dataset, test_dataset = query_splitter.split(dataset, 0.5)

        assert "S1" in train_dataset.artifact_map
        assert len(train_dataset.trace_df) == 1
        assert len(train_dataset.artifact_df) == 2
        assert "T1" in train_dataset.artifact_map or "T2" in train_dataset.artifact_map

        assert "S1" in test_dataset.artifact_map
        assert len(test_dataset.trace_df) == 1
        assert len(test_dataset.artifact_df) == 2
        assert "T1" in train_dataset.artifact_map or "T2" in train_dataset.artifact_map
