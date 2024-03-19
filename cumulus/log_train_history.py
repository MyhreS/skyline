from typing import Dict
import json
import os


def log_train_history(history: Dict, run_id: str):
    path_to_model_train_history = os.path.join("cache", run_id, "train_history.json")
    if not os.path.exists(os.path.dirname(path_to_model_train_history)):
        os.makedirs(os.path.dirname(path_to_model_train_history))
    with open(path_to_model_train_history, "w") as f:
        json.dump(history, f, indent=4)
