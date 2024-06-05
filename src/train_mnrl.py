import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData, SentenceTransformerTrainer, \
    SentenceTransformerTrainingArguments, losses

from tdata.trace_dataset import TraceDataset


def to_mnrl_dataset(dataset: TraceDataset):
    anchors = []
    positives = []
    for i, row in dataset.trace_df.iterrows():
        s_id = row['source']
        t_id = row['target']
        anchors.append(dataset.artifact_map[t_id])
        positives.append(dataset.artifact_map[s_id])

    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives
    })


def train_mnrl(dataset: TraceDataset, n_epochs: int, model_name: str = "all-MiniLM-L6-v2",
               output_path: str = None, **kwargs):
    # Merge to create pairs
    train_trace_dataset, val_trace_dataset = dataset.split(0.1)
    train_dataset = to_mnrl_dataset(train_trace_dataset)
    val_dataset = to_mnrl_dataset(val_trace_dataset)

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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss
    )

    # Train the model
    trainer.train()
    return trainer.model
