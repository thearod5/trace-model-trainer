from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.util import cos_sim

from constants import BATCH_SIZE, DEFAULT_FP16, DEFAULT_ST_MODEL, LEARNING_RATE, N_EPOCHS
from models.itrace_model import ITraceModel, SimilarityMatrix
from readers.trace_dataset import TraceDataset


class STModel(ITraceModel):
    def __init__(self, model_name: str = DEFAULT_ST_MODEL):
        self.model_name = model_name
        self._model = None

    def train(self, train_dataset_map: Dict[str, TraceDataset], losses, output_path, best_metric: str = None, *args, **kwargs) -> None:
        assert isinstance(train_dataset_map, dict)

        has_gpu = torch.cuda.is_available()
        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=output_path,
            # Optional training parameters:
            num_train_epochs=N_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.1,
            fp16=DEFAULT_FP16 and has_gpu,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            # Optional tracking/debugging parameters:
            save_strategy="epoch",
            save_steps=1,
            save_total_limit=2,
            logging_steps=1
        )
        if best_metric:
            args.eval_strategy = "epoch"
            args.eval_steps = 1
            args.load_best_model_at_end = True
            args.metric_for_best_model = best_metric

        trainer = SentenceTransformerTrainer(self.get_model(),
                                             args=args,
                                             train_dataset=train_dataset_map,
                                             loss=losses,
                                             **kwargs)
        trainer.train()
        self._model = trainer.model

    def predict(self, sources: List[str], targets: List[str]) -> SimilarityMatrix:
        """
        Embeds all artifacts and creates predictions from similarity scores.
        :param sources: The source artifacts to compare against the targets.
        :param targets: The target artifacts.
        :return: List of trace predictions containing embedding similarity scores.
        """
        texts = list(set(sources).union(set(targets)))
        embeddings = self.get_model().encode(texts)
        embedding_map = {t: e for t, e in zip(texts, embeddings)}
        source_embeddings = [embedding_map[s] for s in sources]
        target_embeddings = [embedding_map[t] for t in targets]
        similarity_matrix = cos_sim(source_embeddings, target_embeddings)
        return similarity_matrix

    def get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
