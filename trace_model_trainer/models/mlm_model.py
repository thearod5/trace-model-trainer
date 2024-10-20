import os
from typing import Dict, List

import torch
from datasets import Dataset
from sentence_transformers.util import cos_sim
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizerFast, Trainer, \
    TrainingArguments

from constants import BATCH_SIZE, DEFAULT_MLM_MODEL, N_EPOCHS
from models.itrace_model import ITraceModel, SimilarityMatrix
from readers.trace_dataset import TraceDataset

DEFAULT_BLOCK_SIZE = 128


class MLMModel(ITraceModel):
    def __init__(self, model_name: str = DEFAULT_MLM_MODEL, mlm_probability: float = 0.15):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlm_probability = mlm_probability

    def predict(self, sources: List[str], targets: List[str]) -> SimilarityMatrix:
        texts = list(set(sources).union(set(targets)))
        embeddings = self.get_mlm_embeddings(self.model, self.tokenizer, texts)
        embedding_map = {t: e for t, e in zip(texts, embeddings)}

        source_embeddings = [embedding_map[s] for s in sources]
        target_embeddings = [embedding_map[t] for t in targets]
        similarity_matrix = cos_sim(torch.stack(source_embeddings), torch.stack(target_embeddings))
        return similarity_matrix

    def train(self, dataset_map: Dict[str, TraceDataset] | TraceDataset, output_path: str, *args, **kwargs):
        assert isinstance(dataset_map, TraceDataset), "MLM only accepting trace dataset ATM."

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_probability
        )
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=N_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            save_strategy="epoch",
            # evaluation_strategy="epoch",
            # load_best_model_at_end=True,
            # metric_for_best_model="loss",
            logging_dir=os.path.join(output_path, 'logs'),
            no_cuda=~torch.cuda.is_available(),
            logging_steps=1,
        )
        train_dataset = self._to_train_dataset(dataset_map, self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            *args, **kwargs
        )

        trainer.train()

    @staticmethod
    def _to_train_dataset(trace_dataset: TraceDataset, tokenizer: PreTrainedTokenizerFast):
        def preprocess_function(examples):
            return tokenizer(examples["text"])

        dataset = Dataset.from_dict({"text": trace_dataset.artifact_map.values()})

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset.column_names,
        )
        lm_dataset = tokenized_dataset.map(MLMModel.group_texts, batched=True, num_proc=4)
        return lm_dataset

    @staticmethod
    def group_texts(examples, block_size=DEFAULT_BLOCK_SIZE):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    @staticmethod
    def get_mlm_embeddings(model, tokenizer, sentences):
        # Encode the sentences
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Get model outputs (last hidden states)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract the last hidden states (embeddings for each token)
        last_hidden_states = outputs.hidden_states[-1]

        # Compute the mean of all token embeddings (ignoring padding tokens)
        # Use the 'attention_mask' to exclude padding tokens from the averaging
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)  # Avoid division by zero
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings
