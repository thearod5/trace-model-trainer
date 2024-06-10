from experiment.vsm import VSMController
from tdata.trace_dataset import TraceDataset


def create_experiment_dataset(dataset: TraceDataset, min_words: int = 5):
    artifact_df = dataset.artifact_df.copy()

    vsm_controller = VSMController()
    vsm_controller.train(dataset.artifact_map.values())

    artifact_df["content"] = artifact_df["content"].apply(lambda a_content: vsm_controller.get_top_n_words(a_content, min_words))

    return TraceDataset(artifact_df, dataset.trace_df, dataset.layer_df)
