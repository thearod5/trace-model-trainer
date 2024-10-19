from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformerTrainingArguments

from st3traceability.constants import BATCH_SIZE, LEARNING_RATE, N_EPOCHS

DEFAULT_FP16 = True


@dataclass
class TrainingContext:

    @staticmethod
    def create_args(output_path: str, *args, **kwargs) -> SentenceTransformerTrainingArguments:
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
            eval_strategy="epoch",
            eval_steps=1,
            save_strategy="epoch",
            save_steps=1,
            save_total_limit=2,
            logging_steps=1,
            load_best_model_at_end=True,
            *args,
            **kwargs
        )
        return args
