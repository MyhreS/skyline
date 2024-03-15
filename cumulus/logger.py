import datetime
import pandas as pd
import tensorflow as tf
import os
from typing import Dict
import json
import shutil
from contextlib import redirect_stdout

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


class Logger:
    """
    A class which saves the classification model results, the data description, model description etc.
    """

    def __init__(self, run_name: str, clean=False):
        self.path_to_cache = "cache"
        # Add time to the run name (only date and time, not seconds)
        self.name = f"{run_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        self.create_run_log(run_name, clean)

    def create_run_log(self, run_name: str, clean=False):
        # Cache directory check
        if not os.path.exists(self.path_to_cache):
            os.makedirs(self.path_to_cache)

        # Remove directories with the same name (looking away from the date and time part of the name)
        if clean:
            for folder in os.listdir(self.path_to_cache):
                if run_name in folder:
                    shutil.rmtree(os.path.join(self.path_to_cache, folder))

        # Create run directory
        path_to_run = os.path.join(self.path_to_cache, self.name)
        os.makedirs(path_to_run)

        # Create tensorboard directory
        path_to_tensorboard = os.path.join(path_to_run, "tensorboard")
        os.makedirs(path_to_tensorboard)

        # Create model directory
        path_to_model = os.path.join(path_to_run, "model")
        os.makedirs(path_to_model)

        # Copy the preprocessing_config.json file to the run directory
        path_to_preprocessing_config = "cache/datamaker_config.json"
        if os.path.exists(path_to_preprocessing_config):
            shutil.copy(path_to_preprocessing_config, path_to_run)

        # Make tuner directory
        path_to_tuner = os.path.join(path_to_run, "tuner")
        os.makedirs(path_to_tuner)

    def log_model(self, model: tf.keras.Model):
        path_to_model = os.path.join(self.path_to_cache, self.name, "model")
        model.save(path_to_model)

    def log_data_config(self, data_config: Dict):
        # Create a json file and write the data description to it
        path_to_data_config = os.path.join(
            self.path_to_cache, self.name, "data_config.json"
        )
        with open(path_to_data_config, "w") as f:
            json.dump(data_config, f, indent=4)

    def log_model_info(self, model):
        path_to_model_info = os.path.join(
            self.path_to_cache, self.name, "model_info.txt"
        )
        with open(path_to_model_info, "w") as f:
            with redirect_stdout(f):
                model.summary()

    def log_train_history(self, history: Dict):
        path_to_model_train_history = os.path.join(
            self.path_to_cache, self.name, "train_history.json"
        )
        with open(path_to_model_train_history, "w") as f:
            json.dump(history, f, indent=4)

    def log_test_results(self, test_results: dict):
        # Save the confusion matrices to a txt file
        path_to_confusion_matrices = os.path.join(
            self.path_to_cache, self.name, "test_confusion_matrices.txt"
        )
        with open(path_to_confusion_matrices, "w") as f:
            for dataset_name, results in test_results.items():
                # Create an emphasized headline for each dataset
                headline = f"Confusion Matrix for dataset: {dataset_name}"
                f.write(headline + "\n")
                f.write("=" * len(headline) + "\n")  # Underline the headline

                confusion_matrix: pd.DataFrame = results["confusion_matrix"]
                # Convert the confusion matrix to a string
                matrix_str = confusion_matrix.to_string()
                # Find the index of the first newline character to determine where to insert the separator line
                first_newline_idx = matrix_str.find('\n')
                # Insert a line of dashes below the column headers
                matrix_str = matrix_str[:first_newline_idx] + "\n" + "-" * first_newline_idx + matrix_str[first_newline_idx:]

                f.write(matrix_str)
                f.write("\n\n")

        # Now save the other test results without the confusion matrices
        for dataset_name, results in test_results.items():
            results.pop("confusion_matrix")
        path_to_test_results = os.path.join(
            self.path_to_cache, self.name, "test_results.json"
        )
        with open(path_to_test_results, "w") as f:
            json.dump(test_results, f, indent=4)

    def get_tensorboard_path(self):
        return os.path.join(self.path_to_cache, self.name, "tensorboard")

    def get_tuner_path(self):
        return os.path.join(self.path_to_cache, self.name, "tuner")

    def get_run_name(self):
        return self.name
