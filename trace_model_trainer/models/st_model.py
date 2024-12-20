import os
from typing import Dict, List, Type

import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim
from torch import nn

from trace_model_trainer.constants import DEFAULT_FP16, DEFAULT_ST_MODEL, N_EPOCHS
from trace_model_trainer.models.itrace_model import ITraceModel, SimilarityMatrix
from trace_model_trainer.tdata.trace_dataset import TraceDataset


class STModel(ITraceModel):
    def __init__(self, model_name: str = DEFAULT_ST_MODEL, prefix: str = None):
        """
        Creates sentence transformer model with given model and formatter.
        :param model_name:
        :param prefix: Prefix to append to each sentence.
        """
        self.model_name = model_name
        self._model = None
        self.prefix = prefix

    def train(self,
              train_dataset: Dataset | DatasetDict | Dict[str, Dataset],
              losses: Dict[str, nn.Module],
              eval_dataset: Dataset | DatasetDict | Dict[str, Dataset] = None,
              output_path=None,
              args: Dict = None,
              batch_size: int = 8,
              learning_rate: float = 5e-5,
              eval_strategy: str = "epoch",
              eval_steps: int = 1,
              load_best_model_at_end: bool = True,
              metric_for_best_model: str = "loss",
              trainer_class: Type[SentenceTransformerTrainer] = SentenceTransformerTrainer,
              **kwargs) -> SentenceTransformerTrainer:

        args = args or {}

        has_gpu = torch.cuda.is_available()
        print("Learning rate:", learning_rate)
        trainer_args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=output_path,
            # Optional training parameters:
            num_train_epochs=N_EPOCHS,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            fp16=DEFAULT_FP16 and has_gpu,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            # Logging
            logging_dir=os.path.join(output_path, "logs"),
            logging_strategy="epoch",
            logging_steps=1,
            # Optional tracking/debugging parameters:
            save_strategy="epoch",
            save_steps=1,
            save_total_limit=2,
            report_to=None,
            batch_sampler=BatchSamplers.NO_DUPLICATES
        )

        trainer_args.eval_strategy = eval_strategy
        trainer_args.eval_steps = eval_steps
        trainer_args.load_best_model_at_end = load_best_model_at_end
        trainer_args.metric_for_best_model = metric_for_best_model

        for k, v in args.items():
            setattr(trainer_args, k, v)

        if eval_dataset:
            kwargs["eval_dataset"] = eval_dataset

        trainer = trainer_class(self.get_model(),
                                args=trainer_args,
                                train_dataset=train_dataset,
                                loss=losses,
                                **kwargs)

        trainer.train()
        self._model = trainer.model
        return trainer

    def predict(self, sources: List[str], targets: List[str]) -> SimilarityMatrix:
        """
        Embeds all artifacts and creates predictions from similarity scores.
        :param sources: The source artifacts to compare against the targets.
        :param targets: The target artifacts.
        :return: List of trace predictions containing embedding similarity scores.
        """
        texts = list(set(sources).union(set(targets)))
        embeddings = self.get_model().encode(texts, prompt=self.prefix)
        embedding_map = {t: e for t, e in zip(texts, embeddings)}
        source_embeddings = [embedding_map[s] for s in sources]
        target_embeddings = [embedding_map[t] for t in targets]
        similarity_matrix = cos_sim(source_embeddings, target_embeddings)
        return similarity_matrix.tolist()

    def get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _format_dataset(self, dataset: TraceDataset | Dict[str, TraceDataset]):
        if isinstance(dataset, TraceDataset):
            return self._formatter.format(dataset)
        elif isinstance(dataset, dict):
            return {k: self._formatter.format(d) for k, d in dataset.items()}
        else:
            raise Exception(f"Unsupported dataset type: {dataset}")
