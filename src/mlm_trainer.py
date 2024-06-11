from datasets import Dataset
from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from infra.eval import eval_model, print_metrics
from tdata.reader import read_project

if __name__ == "__main__":
    train_project_path = "../res/safa"

    #
    # Training and Evaluation
    #
    eval_dataset = read_project(train_project_path)
    train_dataset = read_project(train_project_path)
    artifact_content = list(train_dataset.artifact_map.values())
    dataset = Dataset.from_dict({"text": artifact_content})

    # Load SentenceTransformer model
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    before_metrics, _ = eval_model(model, eval_dataset, title="Before Training")


    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)


    batch_size = 8
    tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=batch_size)
    split_datasets = tokenized_datasets.train_test_split(test_size=0.1)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Training Args
    training_args = TrainingArguments(
        output_dir="./bert_mlm",
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        logging_dir='./logs',
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=split_datasets["train"].remove_columns(["text"]),
        eval_dataset=split_datasets["test"].remove_columns(["text"]),
    )

    trainer.train()

    after_metrics, _ = eval_model(trainer.model, eval_dataset, title="After Training")

    print_metrics([before_metrics, after_metrics], ["before training", "after training"])
