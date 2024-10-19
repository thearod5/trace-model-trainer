from datasets import Dataset

from st3traceability.formatters.iformatter import IFormatter


class ClassificationFormatter(IFormatter):
    """
    Formats dataset as list of classified pairs of text (source/target) as either traced (1) or not traced (0).
    """

    def format(self, dataset: Dataset) -> Dataset:
        rename_map = {"s_text": "sentence1", "t_text": "sentence2", "label": "label"}
        cols_to_remove = [col for col in dataset.column_names if col not in rename_map.values()]
        return dataset.remove_columns(cols_to_remove).rename_columns(rename_map)
