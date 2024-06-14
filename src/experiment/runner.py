from collections import defaultdict

from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from experiment.factory import create_bootstrapped_links, create_vsm_important_words
from infra.eval import predict_model, print_metrics
from infra.generic_trainer import generic_train
from tdata.reader import read_project
from tdata.trace_dataset import TraceDataset

REFLECTIVE = "reflective"
CONTRASTIVE_TENSION = "contrastive_tension"
VSM_IMPORTANCE_TYPE = "vsm_importance"
BOOTSTRAP_TYPE = "bootstrap"
experiments = {VSM_IMPORTANCE_TYPE, BOOTSTRAP_TYPE}


def run_experiment(eval_project_path: str,
                   model_name: str,
                   loss_name: str = "cosent",
                   training_type=REFLECTIVE,
                   do_train: bool = True,
                   do_eval_before=True):
    metrics = defaultdict(dict)
    model = SentenceTransformer(model_name)
    eval_dataset = read_project(eval_project_path)
    if training_type == VSM_IMPORTANCE_TYPE:
        train_dataset = create_vsm_important_words(read_project(eval_project_path),
                                                   min_words=3,
                                                   threshold=4.0,
                                                   self_links=True)
    elif training_type == BOOTSTRAP_TYPE:
        train_dataset = create_bootstrapped_links(eval_dataset, model)
    elif training_type == CONTRASTIVE_TENSION:
        train_dataset = TraceDataset(eval_dataset.artifact_df,
                                     DataFrame(columns=["source", "target", "label"]),
                                     DataFrame(columns=["source_type", "target_type"]))
    elif training_type == REFLECTIVE:
        eval_dataset, train_dataset = eval_dataset.split(0.25)
        print("Train:", len(train_dataset))
        print("Eval:", len(eval_dataset))
    else:
        raise Exception(f"{training_type} is not one of {experiments}")

    if do_eval_before:
        # Before Training Evaluate
        m1, _ = predict_model(model, eval_dataset)
        metrics["eval"]['no-training'] = m1

    # Training
    if do_train:
        trained_model = generic_train(train_dataset,
                                      loss_name=loss_name,
                                      model=model,
                                      n_epochs=1,
                                      disable_tqdm=False)
        # Eval
        m3, _ = predict_model(trained_model, eval_dataset)
        metrics['eval']['after-training'] = m3
    print_metrics(metrics, order=['train', 'eval'], levels=1)
