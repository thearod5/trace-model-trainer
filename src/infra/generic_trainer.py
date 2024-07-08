import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData, SentenceTransformerTrainer, \
    SentenceTransformerTrainingArguments, losses
from torch.optim import Adam
from transformers import get_scheduler

from constants import OUTPUT_PATH
from tdata.factory import to_dataset
from tdata.trace_dataset import TraceDataset
from utils import clear_memory

loss2function = {
    "contrastive_tension": losses.ContrastiveTensionLossInBatchNegatives,
    "cosent": losses.CoSENTLoss,
    "triplet": losses.TripletLoss,
    "mnrl": losses.MultipleNegativesRankingLoss,
    "mnrl_symetric": losses.MultipleNegativesSymmetricRankingLoss
}

loss2dataset = {
    losses.CoSENTLoss: "float",
    losses.TripletLoss: "triplet",
    losses.MultipleNegativesRankingLoss: "mnrl",
    losses.MultipleNegativesSymmetricRankingLoss: "mnrl",
    losses.ContrastiveTensionLossInBatchNegatives: "contrastive_tension"
}


def generic_train(dataset: TraceDataset,
                  loss_name: str = "triplet",
                  n_epochs: int = 1,
                  val_trace_dataset: TraceDataset = None,
                  model_name: str = "all-MiniLM-L6-v2",
                  model: SentenceTransformer = None,
                  output_path: str = None,
                  batch_size: int = 16,
                  learning_rate=5e-5,
                  warm_up_ration: float = 0.1,
                  **kwargs):
    assert loss_name in loss2function, f"{loss_name} not one of {loss2function.keys()}"
    loss_fnc = loss2function[loss_name]
    dataset_type = loss2dataset[loss_fnc]

    if val_trace_dataset is None:
        train_trace_dataset, val_trace_dataset = dataset.split(0.1)
    if output_path is None:
        output_path = OUTPUT_PATH

    # Merge to create pairs
    train_dataset = to_dataset(dataset, dataset_type)
    val_dataset = to_dataset(val_trace_dataset, dataset_type)

    # Load a pre-trained SentenceTransformer model
    if not model:
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
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warm_up_ration,
        fp16=torch.cuda.is_available(),
        bf16=False,
        eval_strategy="epoch",
        eval_steps=1,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
        logging_strategy="no",
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        **kwargs
    )

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Initialize the loss function and training arguments
    total_steps = len(train_dataset) // batch_size * n_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=total_steps * warm_up_ration,  # 10% of training as warmup
        num_training_steps=total_steps
    )

    # Initialize the loss function
    loss = loss_fnc(model)

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        optimizers=(optimizer, scheduler)
    )

    # Train the model
    trainer.train()
    clear_memory()

    return trainer.model
