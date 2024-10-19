from datasets import Dataset

from st3traceability.formatters.iformatter import IFormatter


class AnchorPositiveFormatter(IFormatter):
    """
    Formats dataset to represents pairs of positively associated strings (source/target)
    """

    def format(self, dataset: Dataset) -> Dataset:
        rename_map = {"s_text": "anchor", "t_text": "positive"}
        dataset = dataset.rename_columns(rename_map)
        cols_to_remove = [col for col in dataset.column_names if col not in rename_map.values()]
        dataset = dataset.remove_columns(cols_to_remove)
        return dataset
