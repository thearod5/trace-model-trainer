import os
from os.path import dirname

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData, SentenceTransformerTrainer, \
    SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import TripletLoss

from training_data import TrainingData


def train_triplet(training_data: TrainingData, export_path: str, model_name: str = "all-MiniLM-L6-v2"):
    train_df = training_data.train_df
    val_df = training_data.val_df

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
    args = SentenceTransformerTrainingArguments(
        output_dir="models/mpnet-base-custom-triplet",
        num_train_epochs=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        bf16=False,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # Create the evaluators
    dev_evaluator = TripletEvaluator(
        anchors=val_df["anchor"],
        positives=val_df["positive"],
        negatives=val_df["negative"],
        name="custom-triplet-dev"
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
    os.makedirs(dirname(export_path), exist_ok=True)
    trainer.model.save_pretrained(export_path)
    print("Saved model to:", export_path)

    return trainer.model

# Note: Ensure that the 'training_data' module provides 'train_df' and 'val_df' dataframes with the columns "anchor", "positive", and "negative".
