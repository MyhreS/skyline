import logging
from enum import Enum
from cumulus import Logger
from cirrus import Data
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def calculate_accuracy(predictions, true_labels):
    correct = 0
    for prediction, true_label in zip(predictions, true_labels):
        if prediction == true_label:
            correct += 1
    return round(correct / len(predictions), 3)


def calculate_confusion_matrix(predictions: List[str], true_labels: List[str], unique_labels: List[str]):
    # Initialize the confusion matrix with zeros
    confusion_matrix = pd.DataFrame(np.zeros((len(unique_labels), len(unique_labels))), index=unique_labels, columns=unique_labels)
    # Populate the confusion matrix
    for true, pred in zip(true_labels, predictions):
        confusion_matrix.loc[true, pred] += 1
    return confusion_matrix


class Evaluater:
    def __init__(
        self,
        model: tf.keras.Model,
        data: Data,
        logger: Logger,
        label_encoding: str,
    ):
        self.model = model
        self.logger = logger
        self.data = data
        self.label_encoding = label_encoding
        self.test_on_datasets()

    def test_on_datasets(self):
        for test_dataset in self.data.dataloader.get_names_of_test_datasets() + [
            "test"
        ]:
            self.test_on_dataset(test_dataset)

    def test_on_dataset(self, dataset_name: str):
        dataset, *_ = self.data.load_it(
            split=dataset_name, label_encoding=self.label_encoding
        )  # This is batched

        predictions = []
        results = self.model.predict(dataset, verbose=1)
        for result in results:
            predictions.append(
                self.data.dataloader.class_encoder.decode_class(
                    result, self.label_encoding
                )
            )

        true_labels = []
        for _, y_batch in dataset:
            for true_label in y_batch:
                true_labels.append(
                    self.data.dataloader.class_encoder.decode_class(
                        true_label, self.label_encoding
                    )
                )

        accuracy = calculate_accuracy(predictions, true_labels)
        print(f"Accuracy on {dataset_name}: {accuracy}")
        confusion_matrix = calculate_confusion_matrix(
            predictions,
            true_labels,
            list(self.data.dataloader.class_encoder.label_class_map.keys()),
        )
        print(confusion_matrix)

        """
        label_class_map looks like this:
        {
            "drone": {
                "integer_encoding": 0,
                "onehot_encoding": [
                    1,
                    0
                ],
                "labels_belonging_to_class": [
                    "electric_quad_drone",
                    "racing_drone",
                    "electric_fixedwing_drone",
                    "petrol_fixedwing_drone"
                ]
            },
            "non-drone": {
                "integer_encoding": 1,
                "onehot_encoding": [
                    0,
                    1
                ],
                "labels_belonging_to_class": [
                    "dvc_non_drone",
                    "animal",
                    "speech",
                    "TUT_dcase",
                    "nature_chernobyl"
                ]
            }
        }
        """
        # Calculate accuracy

        # Evaluate using tf evaluate
        result = self.model.evaluate(dataset)
