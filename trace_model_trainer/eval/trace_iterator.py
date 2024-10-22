from typing import List, Tuple

from trace_model_trainer.readers.trace_dataset import TraceDataset


def trace_iterator(dataset: TraceDataset, empty_ok: bool = False) -> List[Tuple[str, str]]:
    """
    Iterates through each set of traced artifacts in the dataset.
    :param dataset: The dataset whose trace you want to iterate over.
    :param empty_ok: Whether to throw error is iterator is empty.
    :return: List of tuples containing traced artifact ids.
    """
    if len(dataset.layer_df) == 0 and not empty_ok:
        raise Exception("Attempted to retrieve combination of traces, but no matrix defined.")

    payload = []
    for i, layer_row in dataset.layer_df.iterrows():
        source_layer = layer_row["source_type"]
        target_layer = layer_row["target_type"]

        source_artifact_df = dataset.get_by_type(source_layer)
        target_artifact_df = dataset.get_by_type(target_layer)

        source_artifact_ids = source_artifact_df['id']
        target_artifact_ids = target_artifact_df['id']
        payload.append((source_artifact_ids, target_artifact_ids))
    return payload
