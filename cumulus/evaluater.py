import os
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Dict
from .class_decoder import ClassDecoder
from tensorflow.keras.utils import image_dataset_from_directory


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
    ):
        self.model = model
        self.decoder = ClassDecoder(class_label_map)
        self.path_to_images_directory = path_to_images_directory
        self.test_on_datasets()

    def test_on_datasets(self):
        tests = {}
        for test_dataset in get_test_dataset_names(self.path_to_images_directory):
            accuracy, confusion_matrix = self.test_on_dataset(test_dataset)
            tests[test_dataset] = {
                "accuracy": accuracy,
                "confusion_matrix": confusion_matrix,
            }

    def test_on_dataset(self, dataset_name: str):
        print(f"\nLoading dataset {dataset_name}")
        dataset = image_dataset_from_directory(
            os.path.join(self.path_to_images_directory, dataset_name),
            seed=123,
            image_size=(63, 512),
            batch_size=32,
            color_mode="grayscale",
            # label_mode="binary",
        )
        print("Model.evaluating..")
        self.model.evaluate(dataset)

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
        print("Confusion Matrix:")
        print(confusion_matrix)
        return accuracy, confusion_matrix
