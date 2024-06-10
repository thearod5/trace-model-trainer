from collections import defaultdict
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

from experiment.vsm import VSMController
from tdata.trace_dataset import TraceDataset
from tdata.types import TracePrediction
from utils import clear_memory, get_device, scale, t_id_creator


def eval_model(model, dataset: TraceDataset, model_name: str = None, title=None, print_missing_links: bool = False,
               disable_logs: bool = False, include_vsm: bool = True):
    if model_name:
        model = SentenceTransformer(model_name)
    device = get_device(disable_logs=disable_logs)
    model.eval()
    model.to(device)
    trace_predictions = predict_scores(model, dataset, disable_logs=disable_logs)
    metrics = calculate_metrics(trace_predictions)

    if print_missing_links:
        missed_links = [p for p in trace_predictions if p.score < 0.5 and p.label == 1]
        print_links(missed_links)

    if title:
        print(title)
        print(metrics)

    model.to('cpu')
    clear_memory()
    return metrics, trace_predictions


def eval_vsm(dataset: TraceDataset, title: str = None):
    vsm_predictions = get_vsm_predictions(dataset)
    vsm_metrics = calculate_metrics(vsm_predictions)
    if title:
        print(title)
        print(vsm_metrics)
    return vsm_metrics, vsm_predictions


def calculate_metrics(trace_predictions, prefix: str = None):
    ap_scores = calculate_map(trace_predictions)
    predictions = []
    labels = []
    for t in trace_predictions:
        predictions.append(1 if t.score >= 0.5 else 0)
        labels.append(t.label)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics = {
        "map": sum(ap_scores) / len(ap_scores),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "f2": fbeta_score(labels, predictions, beta=2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def calculate_map(trace_predictions):
    query2preds = defaultdict(list)
    for trace in trace_predictions:
        query2preds[trace.target].append(trace)
    ap_scores = []
    for target, predictions in query2preds.items():
        scores = [p.score for p in predictions]
        labels = [p.label for p in predictions]
        ap = average_precision_score(labels, scores)
        if np.isnan(ap):
            continue
        ap_scores.append(ap)
    return ap_scores


def predict_scores(model: SentenceTransformer,
                   dataset: TraceDataset,
                   use_ids: bool = True,
                   disable_logs: bool = False,
                   use_scaling: bool = True) -> List[TracePrediction]:
    content_ids = list(dataset.artifact_map.keys())
    content_set = [f"({a_id}) {a_c}" if use_ids else a_c for a_id, a_c in dataset.artifact_map.items()]
    embeddings = model.encode(content_set, convert_to_tensor=False, show_progress_bar=not disable_logs)
    embedding_map = {a_id: e for a_id, e in zip(content_ids, embeddings)}

    predictions = []
    for source_artifact_ids, target_artifact_ids in dataset.get_layer_iterator():
        source_embeddings = np.stack([embedding_map[s_id] for s_id in source_artifact_ids])
        target_embeddings = np.stack([embedding_map[t_id] for t_id in target_artifact_ids])

        scores = cosine_similarity(source_embeddings, target_embeddings)
        if use_scaling:
            scores = scale(scores)
        for i, s_id in enumerate(source_artifact_ids):
            for j, t_id in enumerate(target_artifact_ids):
                score = scores[i, j]
                label = get_label(dataset, s_id, t_id)

                predictions.append(
                    TracePrediction(
                        source=s_id,
                        target=t_id,
                        label=label,
                        score=score
                    )
                )

    return predictions


def get_vsm_predictions(dataset: TraceDataset, vsm_controller: VSMController = None):
    if vsm_controller is None:
        vsm_controller = VSMController()
        vsm_controller.train(dataset.artifact_map.values())

    predictions = []
    for source_ids, target_ids in dataset.get_layer_iterator():
        source_texts = [dataset.artifact_map[a_id] for a_id in source_ids]
        target_texts = [dataset.artifact_map[a_id] for a_id in target_ids]

        similarity_matrix = vsm_controller.predict(source_texts, target_texts)
        similarity_matrix = scale(similarity_matrix)
        for i, s_id in enumerate(source_ids):
            for j, t_id in enumerate(target_ids):
                label = get_label(dataset, s_id, t_id)
                score = similarity_matrix[i][j]
                predictions.append(
                    TracePrediction(
                        source=s_id,
                        target=t_id,
                        label=label,
                        score=score
                    )
                )
    return predictions


def get_label(dataset: TraceDataset, s_id, t_id):
    trace_id = t_id_creator(source=s_id, target=t_id)
    label = dataset.trace_map[trace_id]['label'] if trace_id in dataset.trace_map else 0
    return label


def print_links(trace_predictions):
    for missed_link in trace_predictions:
        print(f"{missed_link.source} -> {missed_link.target} ({missed_link.score})")


def print_metrics(metrics, metric_names):
    for result_set, set_name in zip(metrics, metric_names):
        print(set_name, result_set)
