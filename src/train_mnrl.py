import torch
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData, SentenceTransformerTrainer, \
    SentenceTransformerTrainingArguments, losses
from torch import Dataset


class MNRLDataset(Dataset):
    def __init__(self, grouped_pairs):
        self.grouped_pairs = grouped_pairs

    def __len__(self):
        return len(self.grouped_pairs)

    def __getitem__(self, idx):
        anchor, positives = self.grouped_pairs[idx][0][0], [p[1] for p in self.grouped_pairs[idx]]
        return {"anchor": anchor, "positive": positives}


def train_mnrl(artifact_df: DataFrame, trace_df: DataFrame, output_path: str, n_epochs: int, model_name: str = "all-MiniLM-L6-v2",
               **kwargs):
    # Merge to create pairs
    pairs = trace_df.merge(artifact_df, left_on='source', right_on='id') \
        .merge(artifact_df, left_on='target', right_on='id', suffixes=('_source', '_target'))

    # Group pairs by anchor
    grouped_pairs = pairs.groupby('source').apply(lambda x: x[['content_source', 'content_target']].values.tolist()).tolist()

    # Custom Torch Dataset

    # Create the dataset
    custom_dataset = MNRLDataset(grouped_pairs)

    # Custom collate function for DataLoader
    def custom_collate_fn(batch):
        anchors = [item["anchor"] for item in batch]
        positives = [item["positive"] for item in batch]
        # Flatten the list of positives and repeat anchors accordingly
        flat_positives = [pos for sublist in positives for pos in sublist]
        flat_anchors = [anchor for anchor, pos_list in zip(anchors, positives) for _ in pos_list]
        return {"anchor": flat_anchors, "positive": flat_positives}

    # Load a pre-trained SentenceTransformer model
    model = SentenceTransformer(
        model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Training on tracing software artifacts.",
        )
    )

    # Define training arguments (adjust as necessary)
    run_name = f"{model_name}-{n_epochs}"
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

    # Initialize the loss function
    loss = losses.MultipleNegativesRankingLoss(model)

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=custom_dataset,
        loss=loss,
        data_collator=custom_collate_fn,
    )

    # Train the model
    trainer.train()
