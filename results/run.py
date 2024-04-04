import sys

sys.path.append("/Users/simonmyhre/workdir/gitdir/skyline")

import json
import pandas as pd
import numpy as np
from cumulus import log_test_results_pretty


RUN = "RUN_9"

"""
Convert the confusion matrix from a dictionary to a DataFrame.
"""
raw_results_path = f"{RUN}/raw_results.json"
with open(raw_results_path, "r") as f:
    raw_results = json.load(f)
for dataset_name, results in raw_results.items():
    confusion_matrix_dict = results["confusion_matrix"]
    confusion_matrix_df = pd.DataFrame.from_dict(
        confusion_matrix_dict, orient="index"
    ).T
    results["confusion_matrix"] = confusion_matrix_df

"""
Get the label map
"""

config_path = f"{RUN}/datamaker_config.json"
with open(config_path, "r") as f:
    config = json.load(f)
label_map = config[0]["label_map"]

log_test_results_pretty(raw_results, label_map, RUN)
