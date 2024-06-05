from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from predict import predict_scores
from tdata.trace_dataset import TraceDataset
from torch_utils import get_device


def eval_model(model, dataset: TraceDataset, title=None):
    device = get_device()
    model.eval()
    model = model.to(device)
    trace_predictions = predict_scores(model, dataset)

    predictions = []
    labels = []
    for t in trace_predictions:
        predictions.append(1 if t.score >= 0.5 else 0)
        labels.append(t.label)

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
