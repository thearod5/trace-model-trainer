import os
from collections import defaultdict
from os.path import dirname

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData, SentenceTransformerTrainer, \
    SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import TripletLoss

from tdata.trace_dataset import TraceDataset
from utils import t_id_creator


def train_triplet(dataset: TraceDataset, n_epochs: int, export_path: str, output_path: str = None,
                  model_name: str = "all-MiniLM-L6-v2", **kwargs):
    if output_path is None:
        output_path = export_path

    os.makedirs(export_path, exist_ok=True)

    train_trace_dataset, val_trace_dataset = dataset.split(0.1)
    train_df = create_triplet_df(train_trace_dataset)
    val_df = create_triplet_df(val_trace_dataset)

    # Load the model
    model = SentenceTransformer(
        model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Training on tracing software artifacts.",
        )
    )

    # Define the triplet loss function
    loss = TripletLoss(model=model)

    # Specify training arguments
    run_name = f"{model_name}-tracing"
    print("Output Dir:", output_path)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        bf16=False,
        eval_strategy="epoch",
        eval_steps=1,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
        logging_steps=100,
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        **kwargs
    )

    # Create the evaluators
    dev_evaluator = TripletEvaluator(
        name="custom-triplet-dev",
        anchors=val_df["anchor"].tolist(),
        positives=val_df["positive"].tolist(),
        negatives=val_df["negative"].tolist(),
        show_progress_bar=False
    )

    # Create the trainer and train the model
    COLS = ["anchor", "positive", "negative"]
    train_dataset = Dataset.from_pandas(train_df[COLS].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[COLS].reset_index(drop=True))
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )

    # Train the model
    trainer.train()

    # Save the best trained model
    if export_path:
        os.makedirs(dirname(export_path), exist_ok=True)
        trainer.model.save_pretrained(export_path)
        print("Saved model to:", export_path)

    return trainer.model


def create_triplet_df(dataset: TraceDataset):
    trace_df = dataset.trace_df
    artifact_map = dataset.artifact_map

    triplet_lookup = create_triplet_lookup(dataset)

    # Create triplets
    triplets = []

    for source_id, lookup in triplet_lookup.items():
        positive_links = pd.Series(lookup['pos'])
        negative_links = pd.Series(lookup['neg']).sample(frac=1)
        for positive_id, negative_id in zip(positive_links, negative_links):
            triplets.append((source_id, positive_id, negative_id))

    triplets = [(artifact_map[a], artifact_map[b], artifact_map[c]) for a, b, c in triplets]
    triplet_df = pd.DataFrame(triplets, columns=['anchor', 'positive', 'negative'])
    return triplet_df


def create_triplet_lookup(dataset: TraceDataset):
    lookup = defaultdict(lambda: {"pos": [], 'neg': []})
    for source_artifact_ids, target_artifact_ids in dataset.get_layer_iterator():
        for s_id in source_artifact_ids:
            for t_id in target_artifact_ids:

                trace_id = t_id_creator(source=s_id, target=t_id)

                if trace_id not in dataset.trace_map:
                    lookup[s_id]['neg'].append(t_id)
                else:
                    lookup[s_id]['pos'].append(t_id)

    return lookup
