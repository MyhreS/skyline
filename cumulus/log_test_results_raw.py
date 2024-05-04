import copy
import json
import os

"""
Function for logging raw test results to a file
"""

def log_raw_results(test_results: dict, run_id: str):
    # Copy rest results to avoid modifying the original dict
    copied_test_results = copy.deepcopy(test_results)

    path_to_raw_results = os.path.join("cache", run_id, "raw_results.json")
    for dataset_name, results in copied_test_results.items():
        results["confusion_matrix"] = results["confusion_matrix"].to_dict()

    with open(path_to_raw_results, "w") as f:
        json.dump(copied_test_results, f, indent=4)
