from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from pandas import DataFrame
from sklearn.metrics import average_precision_score, ndcg_score

from trace_model_trainer.eval.trace_iterator import trace_iterator
from trace_model_trainer.models.itrace_model import ITraceModel
from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.tdata.types import TracePrediction
from trace_model_trainer.utils import create_trace_map, write_json


def eval_model(model: ITraceModel,
               dataset: TraceDataset | Dict[str, TraceDataset],
               save_prediction_path: str = None) -> Tuple[Dict[str, List[TracePrediction]], Dict]:
    if not isinstance(dataset, dict):
        dataset = {"dataset": dataset}

    metrics = {}
    predictions = {}
    for dataset_name, dataset in dataset.items():
        # Create map for easy lookup to attach label later.
        dataset_predictions = compute_model_predictions(model, dataset)

        # Optional - Save predictions
        if save_prediction_path is not None:
            write_json({"predictions": predictions}, save_prediction_path)

        # Calculate metrics
        dataset_metrics = calculate_prediction_metrics(dataset_predictions)
        metrics[dataset_name] = dataset_metrics
        predictions[dataset_name] = dataset_predictions

    return predictions, metrics


def calculate_prediction_metrics(dataset_predictions):
    query2preds = _group_predictions(dataset_predictions)
    query2preds = {query: sorted(preds, key=lambda x: x.score, reverse=True) for query, preds in query2preds.items()}
    map_score, map_lower, map_upper = calculate_map(query2preds)
    mrr_score, mrr_lower, mrr_upper = calculate_mrr(query2preds)
    ndcg_score, ndcg_lower, ndcg_upper = calculate_ndcg(query2preds)
    dataset_metrics = {"map": map_score, "map_lower": map_lower, "map_upper": map_upper,
                       "mrr": mrr_score, "mrr_lower": mrr_lower, "mrr_upper": mrr_upper,
                       "ndcg": ndcg_score, "ndcg_lower": ndcg_lower, "ndcg_upper": ndcg_upper}
    return dataset_metrics


def _group_predictions(predictions: List[TracePrediction]):
    query2preds = defaultdict(list)
    for trace in predictions:
        query2preds[trace.target].append(trace)
    return query2preds


def compute_model_predictions(trace_model: ITraceModel, dataset: TraceDataset):
    trace_map = create_trace_map(dataset.trace_df)

    predictions = []
    for source_ids, target_ids in trace_iterator(dataset):
        source_texts = [dataset.artifact_map[a_id] for a_id in source_ids]
        target_texts = [dataset.artifact_map[a_id] for a_id in target_ids]

        similarity_matrix = trace_model.predict(source_texts, target_texts)
        for s_index, s_id in enumerate(source_ids):
            for t_index, t_id in enumerate(target_ids):
                score = similarity_matrix[s_index][t_index]
                predictions.append(TracePrediction(source=s_id, target=t_id, label=trace_map[s_id].get(t_id, 0), score=score))
    return predictions


def calculate_map(query2preds):
    ap_scores = []
    n_missing_labels = 0
    for target, predictions in query2preds.items():
        scores = [p.score for p in predictions]
        labels = [p.label for p in predictions]
        if 1 not in labels:
            n_missing_labels += 1
            continue
        ap = average_precision_score(labels, scores)
        ap_scores.append(ap)
    print("Queries with no positive labels: ", n_missing_labels)
    if len(ap_scores) == 0:
        print("No queries contained positive lnks.")
        return 0
    return calculate_summary(ap_scores)


def calculate_mrr(query2preds: Dict[str, List[TracePrediction]]):
    reciprocal_ranks = []
    for target, predictions in query2preds.items():
        # Sort predictions by score in descending order
        sorted_preds = sorted(predictions, key=lambda x: x.score, reverse=True)
        query_ranks = []
        for rank, pred in enumerate(sorted_preds, start=1):
            if pred.label == 1:  # If the prediction is correct
                query_ranks.append(1 / (rank - len(query_ranks)))
        reciprocal_ranks.extend(query_ranks)
    if not reciprocal_ranks:
        return 0
    return calculate_summary(reciprocal_ranks)


def calculate_ndcg(query2preds: Dict[str, List[TracePrediction]]):
    ndcg_scores = []
    for target, predictions in query2preds.items():
        scores = [p.score for p in predictions]
        labels = [p.label for p in predictions]
        if 1 not in labels:
            continue
        # Compute NDCG for this target
        ndcg = ndcg_score([labels], [scores])
        ndcg_scores.append(ndcg)
    if not ndcg_scores:
        return 0, 0, 0
    return calculate_summary(ndcg_scores)


def aggregate_metrics(metrics: List[Dict], exclude_group: str = "seed") -> DataFrame:
    """
    Takes list of metrics and computes the average value for each metric.
    :param metrics: List of metrics.
    :return: Map of metric name to average value.
    """
    group_cols = [k for k, v in metrics[0].items() if isinstance(v, str) and k != exclude_group]
    value_cols = [k for k, v in metrics[0].items() if isinstance(v, float) or isinstance(v, int)]
    metric_df = DataFrame(metrics)

    entries = []
    for group, group_df in metric_df.groupby(group_cols):
        value_dict = {v: group_df[v].mean() for v in value_cols}
        key_dict = {k: v for k, v in zip(group_cols, group)}
        entry = {**key_dict, **value_dict}
        entries.append(entry)

    return DataFrame(entries)


def create_retrieval_queries(trace_dataset: TraceDataset):
    target_queries = {}
    trace_map = create_trace_map(trace_dataset.trace_df)
    for source_artifact_ids, target_artifact_ids in trace_iterator(trace_dataset):
        for s_id in source_artifact_ids:
            for t_id in target_artifact_ids:
                if t_id not in target_queries:
                    target_queries[t_id] = {"positive": [], "negative": []}
                label = trace_map[s_id].get(t_id, 0)
                if label == 0:
                    target_queries[t_id]["negative"].append(s_id)
                else:
                    target_queries[t_id]['positive'].append(s_id)

    queries = [{"query": k, **v} for k, v in target_queries.items() if len(v["positive"]) > 0]
    assert len(queries) > 0
    return queries


def calculate_summary(data: List[float]):
    # Calculate the mean and standard deviation
    mean_accuracy = np.mean(data)
    lower_bound = min(data)
    upper_bound = max(data)
    return mean_accuracy, lower_bound, upper_bound
