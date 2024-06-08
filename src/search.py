import os.path
from itertools import product

from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from constants import OUTPUT_PATH
from infra.eval import eval_model
from infra.generic_trainer import generic_train
from tdata.reader import read_project


def search(train_dataset_path: str, test_dataset_path: str, iterable, baseline_model: SentenceTransformer = None,
           **kwargs) -> DataFrame:
    train_dataset = read_project(train_dataset_path)
    test_dataset = read_project(test_dataset_path)
    entries = []

    if baseline_model:
        metrics, _ = eval_model(baseline_model, test_dataset)
        entries.append(metrics)

    for iterable_kwargs in iterable:
        # Train
        trained_model = generic_train(train_dataset,
                                      output_path=OUTPUT_PATH,
                                      **kwargs,
                                      **iterable_kwargs)

        metrics, predictions = eval_model(trained_model, test_dataset)
        entries.append({**iterable_kwargs, **metrics})
    return DataFrame(entries)


def expand_dict_combinations(input_dict):
    # Extract keys and list of values
    keys = input_dict.keys()
    values = (input_dict[key] for key in keys)

    # Compute the Cartesian product of the lists of values
    combinations = product(*values)

    # Convert each combination into a dictionary
    result = [dict(zip(keys, combination)) for combination in combinations]
    return result


if __name__ == "__main__":
    train_project_path = "../res/vsm_experiment"
    eval_project_path = "../res/safa"
    loss_func_name = "mnrl_symetric"
    model_name = "all-MiniLM-L6-v2"
    best_metric_name = "map"

    options = expand_dict_combinations({"loss_name": ['mnrl_symetric', 'triplet']})

    #
    # Search
    #

    model = SentenceTransformer(model_name)
    results_df = search(train_project_path,
                        train_project_path,
                        options,
                        baseline_model=model,
                        disable_tqdm=True)
    results_df.to_csv(os.path.expanduser("~/desktop/results.csv"), index=False)
    print(results_df.sort_values(by=[best_metric_name], ascending=False))
