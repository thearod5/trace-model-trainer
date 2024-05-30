import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from predict import predict_scores
from torch_utils import get_device
from trace_dataset import TraceDataset


def eval_model(model, dataset: TraceDataset, df, title=None):
    device = get_device()
    model.eval()
    model = model.to(device)
    labels = [l for _, _, l in dataset]
    with torch.no_grad():
        scores = predict_scores(model, dataset.get_prediction_payload())

    predictions = [1 if score >= 0.5 else 0 for score in scores]

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    metrics = {
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }
    if title:
        print(title)
    print(metrics)
    return metrics
