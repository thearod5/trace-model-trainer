from datasets import load_dataset


def load_traceability_dataset(dataset_name: str, split: str = "train"):
    """Loads traceability training dataset"""
    return load_dataset(dataset_name, split)[split]
