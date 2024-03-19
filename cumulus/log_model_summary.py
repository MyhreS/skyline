from contextlib import redirect_stdout
import os


def log_model_summary(model, run_id: str):
    path_to_model_info = os.path.join("cache", run_id, "model_summary.txt")
    if not os.path.exists(os.path.dirname(path_to_model_info)):
        os.makedirs(os.path.dirname(path_to_model_info))

    with open(path_to_model_info, "w") as f:
        with redirect_stdout(f):
            model.summary()
