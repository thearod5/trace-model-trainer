import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import minmax_scale

from trace_model_trainer.eval.trace_iterator import trace_iterator
from trace_model_trainer.models.st_model import STModel
from trace_model_trainer.models.vsm_model import VSMModel
from trace_model_trainer.tdata.trace_dataset import TraceDataset


def create_vsm_dataset(dataset: TraceDataset):
    samples = []
    vsm_controller = VSMModel()
    st_model = STModel("all-MiniLM-L6-v2")
    vsm_controller.train({"train": dataset})

    for source_ids, target_ids in trace_iterator(dataset):
        source_texts = [dataset.artifact_map[s_id] for s_id in source_ids]
        target_texts = [dataset.artifact_map[t_id] for t_id in target_ids]
        vsm_matrix = minmax_scale(vsm_controller.predict(source_texts, target_texts))
        st_matrix = minmax_scale(st_model.predict(source_texts, target_texts))

        for s_index, s_text in enumerate(source_texts):
            for t_index, t_text in enumerate(target_texts):
                vsm_score = vsm_matrix[s_index][t_index]
                st_score = st_matrix[s_index][t_index]
                samples.append({
                    "sentence1": s_text,
                    "sentence2": t_text,
                    "label": (vsm_score + st_score) / 2
                })

    dataset = Dataset.from_dict(pd.DataFrame(samples))
    return dataset
