import os
from typing import List

import numpy as np
from nltk import pos_tag, word_tokenize

from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.transforms.augmentation import STOP_WORDS


def filter_nouns(texts) -> List[str]:
    ALLOWED = ["CD", "FW", "LS", "NN", "NNS", "NNP", "NNPS", "SYM", "VB", "VBD", "VBG", "VBN", "VGP", "VBZ", "PRP"]

    transformed = []
    for text in texts:
        allowed_words = [word.lower() for word, tag in pos_tag(word_tokenize(text)) if
                         tag in ALLOWED and word.lower() not in STOP_WORDS]
        transformed.append(" ".join(allowed_words))

    return transformed


def main():
    p1 = os.path.expanduser("~/projects/trace-model-trainer/res/test")
    p2 = "thearod5/CCHIT"
    dataset = load_traceability_dataset(p2)

    np.random.seed(42)

    st_model = STModel("all-MiniLM-L6-v2")
    _, metrics = eval_model(st_model, dataset)
    print("Metrics:\n", metrics)

    dataset.artifact_df["content"] = filter_nouns(dataset.artifact_df["content"].tolist())
    _, metrics = eval_model(st_model, TraceDataset(
        artifact_df=dataset.artifact_df, trace_df=dataset.trace_df, layer_df=dataset.layer_df
    ))
    print("Metrics:\n", metrics)


if __name__ == '__main__':
    main()
