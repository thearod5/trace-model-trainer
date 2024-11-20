import os

from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.poolers.self_attention_pooler import SelfAttentionPooler
from trace_model_trainer.tdata.loader import load_traceability_dataset


def main():
    # Dataset
    dataset_path = os.path.expanduser("~/projects/trace-model-trainer/res/test")
    dataset = load_traceability_dataset(dataset_path)
    artifacts = list(dataset.artifact_map.values())

    # Model
    st_model = STModel("all-MiniLM-L6-v2")
    model = st_model.get_model()
    model.pooling_layer = SelfAttentionPooler(model.get_sentence_embedding_dimension())

    # Encoding
    k = 1
    print(artifacts[:k])
    tokens = model.tokenizer(artifacts[:k])['input_ids'][0]
    num2word = {v: k for k, v in model.tokenizer.vocab.items()}
    print({i: num2word[t] for i, t in enumerate(tokens)})

    model.encode(artifacts[:k])


if __name__ == '__main__':
    main()
