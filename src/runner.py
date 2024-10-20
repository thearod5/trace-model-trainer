from experiment.runner import run_experiment
from infra.eval import run_eval
from infra.generic_trainer import loss2function
from infra.mlm_trainer import run_mlm
from infra.search import run_search
from utils import get_or_prompt

AVAIL_MODELS = ["all-MiniLM-L6-v2", "thearod5/pl-bert-siamese-encoder", "all-roberta-large-v1"]

tools = {
    "train_on_doc": {
        "func": train_on_doc,
        "args": ["TRAIN_PROJECT_PATH", "EVAL_PROJECT_PATH", "MODEL_NAME"]
    },
    "vsm_for_importance": {
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
    "mlm": {
        "func": run_mlm,
        "args": ["TRAIN_PROJECT_PATH", "EVAL_PROJECT_PATH"]
    }
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
    "MIN_WORDS": {
        "prompt": "Minimum number of words required to apply vsm transformation.",
        "type_converter": int
    },
    "VSM_THRESHOLD": {
        "prompt": "VSM threshold to select important words for artifact.",
        "type_converter": float
    }

}

if __name__ == "__main__":
    tool_name = get_or_prompt("TOOL", f"Select Tool", options=tools.keys())

    tool_definition = tools[tool_name.lower()]
    tool_func = tool_definition["func"]
    tool_args = [get_or_prompt(a, **arg_prompts[a]) for a in tool_definition["args"]]
    tool_func(*tool_args)
