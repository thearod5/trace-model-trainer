import os

from search import expand_dict_combinations, search

if __name__ == "__main__":
    train_project_path = "../res/vsm_experiment"
    eval_project_path = "../res/safa"
    loss_func_name = "mnrl_symetric"
    models = ["all-MiniLM-L6-v2"]
    best_metric_name = "map"

    options = expand_dict_combinations({"loss_name": ['mnrl_symetric']})

    #
    # Search
    #

    results_df = search(train_dataset_path=train_project_path,
                        test_dataset_path=eval_project_path,
                        models=models,
                        options=options,
                        disable_logs=True)
    results_df.to_csv(os.path.expanduser("~/desktop/results.csv"), index=False)
    print(results_df.sort_values(by=[best_metric_name], ascending=False))
