import os.path

import pandas as pd

from trace_model_trainer.tdata.exporter import TraceDatasetExporter
from trace_model_trainer.tdata.trace_dataset import TraceDataset

"""
commit_df: commit_id,commit_time,diff,files,summary
link_df: commit_id,issue_id
issue_df: closed_at, created_at,issue_comments,issue_desc,issue_id
"""
COMMIT_TYPE = "Commit"
ISSUE_TYPE = "Issue"
DATASET_PATH = os.path.expanduser("~/desktop/safa/datasets")


def main(dataset_path: str):
    link_df = pd.read_csv(os.path.join(dataset_path, 'clean_link.csv')).drop("Unnamed: 0", axis=1)
    commit_df = pd.read_csv(os.path.join(dataset_path, "clean_commit.csv")).drop("Unnamed: 0", axis=1)
    issue_df = pd.read_csv(os.path.join(dataset_path, "clean_issue.csv")).drop("Unnamed: 0", axis=1)

    # Clean Data Frames
    link_df["issue_id"] = link_df["issue_id"].apply(lambda t: str(round(float(t))))

    commit_df["commit_id"] = commit_df["commit_id"].astype(str)
    issue_df["issue_id"] = issue_df["issue_id"].apply(lambda t: str(round(float(t))))

    # Create Data Frames
    trace_df = link_df.rename({"commit_id": "source", "issue_id": "target"}, axis=1)
    trace_df["label"] = [1] * len(trace_df)

    artifact_df = create_artifact_df(commit_df, issue_df)
    layer_df = pd.DataFrame([{"source_type": COMMIT_TYPE, "target_type": ISSUE_TYPE}])

    # Create dataset
    dataset = TraceDataset(
        artifact_df=artifact_df,
        trace_df=trace_df,
        layer_df=layer_df
    )

    export_path = os.path.join(DATASET_PATH, os.path.basename(dataset_path))
    TraceDatasetExporter.export(dataset=dataset, export_dir=export_path)
    print("Exported:", export_path)


def create_artifact_df(commit_df, issue_df):
    commit_artifacts = []
    for commit_row in commit_df.itertuples():
        commit_artifact = {
            "id": commit_row.commit_id,
            "content": create_commit_artifact(commit_row),
            "summary": None,
            "layer": COMMIT_TYPE
        }
        commit_artifacts.append(commit_artifact)

    issue_artifacts = []
    for issue_row in issue_df.itertuples():
        issue_artifact = {
            "id": issue_row.issue_id,
            "content": create_issue_artifact(issue_row),
            "summary": None,
            "layer": ISSUE_TYPE
        }
        issue_artifacts.append(issue_artifact)
    return pd.DataFrame(commit_artifacts + issue_artifacts)


def create_commit_artifact(row):
    items = ["# Summary\n" + row.summary, "# Diff\n" + row.diff, "# Files\n" + row.files]
    return "\n\n".join(items).strip()


def create_issue_artifact(row):
    items = []
    if isinstance(row.issue_desc, str):
        items.append("# Description\n" + row.issue_desc)

    if isinstance(row.issue_comments, str):
        items.append("# Comments\n" + row.issue_comments)
    return "\n\n".join(items).strip()


if __name__ == '__main__':
    FOLDER_PATH = os.path.expanduser("~/downloads/TBERT git projects")
    DATASETS = ["pallets/flask", "keras-team/keras", "dbcli/pgcli"]
    DATASET_PATHS = [os.path.join(FOLDER_PATH, d) for d in DATASETS]
    main(DATASET_PATHS[0])
    main(DATASET_PATHS[1])
    main(DATASET_PATHS[2])
