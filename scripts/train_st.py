import os.path

import numpy as np
from sentence_transformers.losses import AnglELoss, CoSENTLoss

from trace_model_trainer.eval.splitters.splitter_factory import SplitterFactory
from trace_model_trainer.eval.utils import eval_model
from trace_model_trainer.formatters.classification_formatter import ClassificationFormatter
from trace_model_trainer.loss.custom_cosine_loss import CustomCosineEvaluator
from trace_model_trainer.models.st.balanced_trainer import BalancedTrainer
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.poolers.attention_pooler import AttentionPooler
from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.transforms.vsm_dataset import create_vsm_dataset
from trace_model_trainer.utils import clear_memory


def main():
    np.random.seed(42)
    p0, split0, split1 = (os.path.expanduser("~/projects/trace-model-trainer/res/test"), 0.6, 0.5)
    # p1, split0, split1 = ("thearod5/cchit", 0.2, 0.5)
    dataset = load_traceability_dataset(p0)
    splitter = SplitterFactory.QUERY.create(group_col="target")
    test_output_path = os.path.expanduser("~/projects/trace-model-trainer/output/attention_pooler")

    train_dataset, test_dataset = splitter.split(dataset, train_size=split0)
    train_dataset, val_dataset = splitter.split(train_dataset, train_size=split1)

    vsm_dataset = create_vsm_dataset(dataset)

    st_model = STModel("all-MiniLM-L6-v2")

    model = st_model.get_model()
    model.pooling_layer = AttentionPooler(model.get_sentence_embedding_dimension())

    _, before_metrics = eval_model(st_model, test_dataset)
    print(before_metrics)

    # Train
    vsm_loss = CoSENTLoss(st_model.get_model())
    loss = AnglELoss(st_model.get_model())

    st_model.train(
        train_dataset={
            "train_data": ClassificationFormatter().format(train_dataset),
            "vsm": vsm_dataset
        },
        losses={
            "train_data": loss,
            "vsm": vsm_loss
        },
        output_path=os.path.join(test_output_path, "model"),
        trainer_class=BalancedTrainer,
        evaluator=CustomCosineEvaluator(val_dataset),
        batch_size=4,
        learning_rate=5e-6,
        args={
            "num_train_epochs": 2,
            "enable_full_determinism": True,
            "seed": 42,
            "eval_strategy": "epoch",
            "eval_steps": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_map",
            "greater_is_better": True,
        }
    )

    # Post-Training Evaluation
    _, after_metrics = eval_model(st_model, test_dataset)
    print(before_metrics)
    print(after_metrics)

    # Logging

    del st_model
    st_model = None
    clear_memory()


if __name__ == '__main__':
    main()
