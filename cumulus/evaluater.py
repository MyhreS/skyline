import os
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Dict
from .class_decoder import ClassDecoder
from tensorflow.keras.utils import image_dataset_from_directory
from .log_test_results_pretty import log_test_results_pretty
from .log_test_results_raw import log_raw_results


def get_test_dataset_names(path_to_data):
    paths_in_data = os.listdir(path_to_data)
    return [path for path in paths_in_data if path.startswith("test")]


def calculate_accuracy(predictions, true_labels):
    correct = 0
    for prediction, true_label in zip(predictions, true_labels):
        if prediction == true_label:
            correct += 1
    return round(correct / len(predictions), 3)


def calculate_confusion_matrix(
    predictions: List[str], true_labels: List[str], unique_labels: List[str]
):
    # Initialize the confusion matrix with zeros
    confusion_matrix = pd.DataFrame(
        np.zeros((len(unique_labels), len(unique_labels))),
        index=unique_labels,
        columns=unique_labels,
    )
    # Populate the confusion matrix
    for true, pred in zip(true_labels, predictions):
        confusion_matrix.loc[true, pred] += 1

    confusion_matrix.index.name = "True Class"
    confusion_matrix.columns.name = "Predicted Class"
    return confusion_matrix


class Evaluater:
    def __init__(
        self,
        model: tf.keras.Model,
        class_label_map: Dict[str, List[str]],
        path_to_images_directory: str,
        label_mode: str,
        run_id: str,
    ):
        """
        Class for evaluating a model on a the test datasets.
        Args:
            model: A trained model
            class_label_map: A dictionary like: {"class_1":["label_1", ..], "class_2":[..]}.
            path_to_images_directory: The path to the directory containing the test datasets.
            label_mode: A string specifying the label mode. Can be either "binary" or "categorical".
        """
        self.model = model
        self.class_label_map = class_label_map
        self.decoder = ClassDecoder(class_label_map)
        self.path_to_images_directory = path_to_images_directory
        self.label_mode = label_mode
        self.run_id = run_id
        self.test_on_datasets()

    def test_on_datasets(self):
        tests = {}
        for test_dataset in get_test_dataset_names(self.path_to_images_directory):
            accuracy, evaluate_loss, evaluate_accuracy, confusion_matrix = (
                self.test_on_dataset(test_dataset)
            )
            tests[test_dataset] = {
                "accuracy": accuracy,
                "evaluate_loss": evaluate_loss,
                "evaluate_accuracy": evaluate_accuracy,
                "confusion_matrix": confusion_matrix,
            }
        log_raw_results(tests, self.run_id)
        log_test_results_pretty(tests, self.class_label_map, self.run_id)

    def test_on_dataset(self, dataset_name: str):
        print(f"\nLoading dataset {dataset_name}")
        dataset = image_dataset_from_directory(
            os.path.join(self.path_to_images_directory, dataset_name),
            seed=123,
            image_size=(63, 512),
            batch_size=32,
            color_mode="grayscale",
            label_mode=self.label_mode,
        )
        print("Model.evaluating..")
        evaluate_results = self.model.evaluate(dataset)
        evaluate_accuracy = evaluate_results[1]
        evaluate_loss = evaluate_results[0]

        print("Predicting ..")
        predictions = []
        results = self.model.predict(dataset, verbose=1)
        for result in results:
            predictions.append(self.decoder.decode(result))

        true_labels = []
        for _, y_batch in dataset:
            for true_label in y_batch:
                true_labels.append(self.decoder.decode(true_label.numpy()))

        accuracy = calculate_accuracy(predictions, true_labels)
        confusion_matrix: pd.DataFrame = calculate_confusion_matrix(
            predictions,
            true_labels,
            list(self.decoder.label_class_map.keys()),
        )
        print(f"Accuracy: {accuracy}")
        print(f"Evaluate accuracy: {evaluate_accuracy}")
        print(f"Evaluate loss: {evaluate_loss}")
        print("Confusion Matrix:")
        print(confusion_matrix)
        return accuracy, evaluate_loss, evaluate_accuracy, confusion_matrix
