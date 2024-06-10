from experiment.runner import run_experiment
from utils import get_or_prompt

tools = {
    "experiment": {
        "func": run_experiment,
        "args": ["EVAL_PROJECT_PATH", "MODEL_NAME"]
    }
}

arg_prompts = {
    "EVAL_PROJECT_PATH": {
        "prompt": "Eval Project Path"
    },
    "MODEL_NAME": {
        "prompt": "Model Name",
        "options": ["all-MiniLM-L6-v2", "thearod5/pl-bert-siamese-encoder", "all-roberta-large-v1"]
    }
}

if __name__ == "__main__":
    tool_name = get_or_prompt("TOOL", f"Select Tool", options=tools.keys())

    tool_definition = tools[tool_name.lower()]
    tool_func = tool_definition["func"]
    tool_args = [get_or_prompt(a, **arg_prompts[a]) for a in tool_definition["args"]]
    tool_func(*tool_args)
