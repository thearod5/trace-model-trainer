from eval import run_eval
from experiment.runner import run_experiment
from infra.generic_trainer import loss2function
from search import run_search
from utils import get_or_prompt

AVAIL_MODELS = ["all-MiniLM-L6-v2", "thearod5/pl-bert-siamese-encoder", "all-roberta-large-v1"]

tools = {
    "experiment": {
        "func": run_experiment,
        "args": ["EVAL_PROJECT_PATH", "MODEL_NAME"]
    },
    "eval": {
        "func": run_eval,
        "args": ["EVAL_PROJECT_PATH", "MODEL_NAME"]
    },
    "search": {
        "func": run_search,
        "args": ["TRAIN_PROJECT_PATH", "EVAL_PROJECT_PATH", "LOSS_FUNC_NAME", "MODELS"]
    },
}

arg_prompts = {
    "TRAIN_PROJECT_PATH": {
        "prompt": "Train Project Path"
    },
    "LOSS_FUNC_NAME": {
        "prompt": "Loss Functions",
        "options": loss2function.keys(),
        "type_converter": lambda v: v.split(",")
    },
    "MODELS": {
        "prompt": "List of models",
        "options": AVAIL_MODELS,
        "type_converter": lambda v: v.split(",")
    },
    "EVAL_PROJECT_PATH": {
        "prompt": "Eval Project Path"
    },
    "MODEL_NAME": {
        "prompt": "Model Name",
        "options": AVAIL_MODELS
    },

}

if __name__ == "__main__":
    tool_name = get_or_prompt("TOOL", f"Select Tool", options=tools.keys())

    tool_definition = tools[tool_name.lower()]
    tool_func = tool_definition["func"]
    tool_args = [get_or_prompt(a, **arg_prompts[a]) for a in tool_definition["args"]]
    tool_func(*tool_args)
