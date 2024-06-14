from collections import defaultdict
from typing import List

from tdata.types import TracePrediction


def group_preds(predictions: List[TracePrediction], key_name="source"):
    key2group = defaultdict(list)

    for prediction in predictions:
        key_value = getattr(prediction, key_name)
        key2group[key_value].append(prediction)
    return key2group


def select_top_candidates(predictions: List[TracePrediction], key_name="source"):
    key2preds = group_preds(predictions, key_name=key_name)
    key2preds = {k: sorted(v, key=lambda p: p.score, reverse=True) for k, v in key2preds.items()}

    selected = []
    for k, v in key2preds.items():
        selected.append(v[0])

    keys_missing = [k for k, v in key2preds.items() if all(p.label == 0 for p in v)]
    keys_wrong = [k for k, v in key2preds.items() if v[0].label == 0]
    keys_wrong = set(keys_wrong).difference(keys_missing)
    print("Percent Wrong:", len(keys_wrong) / len(key2preds))
    return selected
# source = 0.41
# target = 0.45
# top_n

# target = 42% wrong
# source = 37% wrong
