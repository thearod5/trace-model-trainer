import os

import pandas as pd

from experiment.vsm import VSMController
from tdata.reader import read_project

if __name__ == "__main__":
    DATA_PATH = os.path.expanduser("~/desktop/safa/datasets")
    project_name = "safa/source"
    project_path = os.path.join(DATA_PATH, project_name)
    project = read_project(project_path)

    vsm_controller = VSMController()
    vsm_controller.train(project.artifact_map.values())

    transformed_artifacts = []
    traces = []

    MIN_WORDS = 5
    for a_id, a_content in project.artifact_map.items():
        transformed_a_id = f"{a_id}_transformed"
        transformed_content = vsm_controller.get_top_n_words(a_content, MIN_WORDS)
        transformed_artifacts.append({
            "id": transformed_a_id,
            "content": transformed_content
        })
        traces.append({
            "source": a_id,
            "target": transformed_a_id
        })

    trace_df = pd.DataFrame(traces)
    transformed_artifact_df = pd.DataFrame(transformed_artifacts)
    EXPORT_PATH = os.path.expanduser("~/projects/trace-model-trainer/res/vsm_experiment")
    os.makedirs(EXPORT_PATH, exist_ok=True)

    project.artifact_df.to_csv(os.path.join(EXPORT_PATH, "source.csv"), index=False)
    transformed_artifact_df.to_csv(os.path.join(EXPORT_PATH, "target.csv"), index=False)
    trace_df.to_csv(os.path.join(EXPORT_PATH, "traces.csv"), index=False)
