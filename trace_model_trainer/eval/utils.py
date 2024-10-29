from collections import defaultdict
from typing import Dict, List, Tuple

from pandas import DataFrame
from sklearn.metrics import average_precision_score, ndcg_score

from trace_model_trainer.eval.trace_iterator import trace_iterator
from trace_model_trainer.models.itrace_model import ITraceModel
from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.tdata.types import TracePrediction
from trace_model_trainer.utils import create_source2targets, write_json


def eval_model(model: ITraceModel, dataset: TraceDataset, save_prediction_path: str = None) -> Tuple[List[TracePrediction], Dict]:
    # Create map for easy lookup to attach label later.
    source2target = create_source2targets(dataset.trace_df)
    predictions = compute_model_predictions(model, dataset)

    # Optional - Save predictions
    if save_prediction_path is not None:
        write_json({"predictions": predictions}, save_prediction_path)

    # Add labels to predictions
    for pred in predictions:
        label = source2target[pred.source].get(pred.target, 0)
        pred.label = label

    # Calculate metrics
    query2preds = _group_predictions(predictions)
    map_score = calculate_map(query2preds)
    mrr_score = calculate_mrr(query2preds)
    ndcg_score = calculate_ndcg(query2preds)
    metrics = {"map": map_score, "mrr": mrr_score, "ndcg": ndcg_score}
    return predictions, metrics


def _group_predictions(predictions: List[TracePrediction]):
    query2preds = defaultdict(list)
    for trace in predictions:
        query2preds[trace.target].append(trace)
    return query2preds


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
    return sum(ap_scores) / len(ap_scores)


def calculate_mrr(query2preds: Dict[str, List[TracePrediction]]) -> float:
    reciprocal_ranks = []
    for target, predictions in query2preds.items():
        # Sort predictions by score in descending order
        sorted_preds = sorted(predictions, key=lambda x: x.score, reverse=True)
        for rank, pred in enumerate(sorted_preds, start=1):
            if pred.label == 1:  # If the prediction is correct
                reciprocal_ranks.append(1 / rank)
                break
    if not reciprocal_ranks:
        return 0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


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
        return 0
    return sum(ndcg_scores) / len(ndcg_scores)


def compute_model_predictions(trace_model: ITraceModel, dataset: TraceDataset):
    predictions = []
    for source_ids, target_ids in trace_iterator(dataset):
        source_texts = [dataset.artifact_map[a_id] for a_id in source_ids]
        target_texts = [dataset.artifact_map[a_id] for a_id in target_ids]

        similarity_matrix = trace_model.predict(source_texts, target_texts)
        for s_index, s_id in enumerate(source_ids):
            for t_index, t_id in enumerate(target_ids):
                score = similarity_matrix[s_index][t_index]
                predictions.append(TracePrediction(source=s_id, target=t_id, label=None, score=score))
    return predictions


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
    trace_map = create_source2targets(trace_dataset.trace_df)
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
