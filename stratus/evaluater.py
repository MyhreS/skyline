import logging
from enum import Enum
from cumulus import Logger
from cirrus import Data
import tensorflow as tf
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


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
        self.run_test()

    def run_test(self):
        test_dataset = self.data.load_it(
            split="test", label_encoding=self.label_encoding
        )
        self.model.evaluate(test_dataset)

        # Create confusion matrix
        actual_labels = []
        all_predictions = []

        for files, labels in test_dataset:
            predictions = self.model.predict(files, verbose=0)
            predicted_labels = np.argmax(
                predictions, axis=1
            )  # Handling a batch of predictions
            if self.label_encoding == "one_hot":
                actual_labels.extend(
                    np.argmax(labels.numpy(), axis=1)
                )  # Handling a batch of labels
            else:
                actual_labels.extend(labels.numpy())

            all_predictions.extend(predicted_labels)

        conf_matrix = tf.math.confusion_matrix(actual_labels, all_predictions)
        print(conf_matrix)


# class Evaluater:
#     def __init__(
#         self,
#         model: tf.keras.Model,
#         data: Data,
#         logger: Logger,
#         label_encoding: str,
#     ):
#         self.test_dataset_names = data.dataloader.get_names_of_test_datasets()
#         self.model = model
#         self.logger = logger
#         self.data = data
#         self.label_encoding = label_encoding
#         self.label_map = self._make_map()
#         print(self.label_map)
#         formatted_index = [
#             test_dataset_name.replace("_", " ").lower()
#             for test_dataset_name in self.test_dataset_names
#         ]
#         self.confusion_matrix = pd.DataFrame(
#             np.zeros((len(self.test_dataset_names), len(self.test_dataset_names))),
#             columns=self.test_dataset_names,
#             index=formatted_index,
#         )
#         self.confusion_matrix.index.name = "Labels"
#         self.confusion_matrix.columns.name = "Datasets"

#     def _make_map(self):
#         label_map = {}
#         for test_dataset_name in self.test_dataset_names:
#             dataset, _ = self.data.load_it(
#                 split=test_dataset_name, label_encoding=self.label_encoding
#             )
#             for _, labels in dataset:
#                 formatted_name = test_dataset_name.replace("_", " ").lower()
#                 if self.label_encoding == "one_hot":
#                     label_map[np.argmax(labels[0].numpy())] = formatted_name
#                 else:
#                     label_map[labels[0].numpy()] = formatted_name
#                 break
#         return label_map

#     def evaluate(self):
#         logging.info("Evaluating model")
#         accuracies = {}

#         for test_dataset_name in self.test_dataset_names:
#             accuracies[test_dataset_name] = self._predict_on_dataset(test_dataset_name)

#         avg_accuracy = sum(accuracies.values()) / len(accuracies)
#         print(f"Average accuracy: {avg_accuracy}")
#         accuracies["average"] = avg_accuracy
#         self.logger.save_model_test_accuracy(accuracies)
#         self.confusion_matrix = self.confusion_matrix.astype(int)
#         print(self.confusion_matrix)
#         self.logger.save_model_test_confusion_matrix(self.confusion_matrix)

#     def _predict_on_dataset(self, test_dataset_name: str):
#         logging.info(f"Evaluating {test_dataset_name} dataset")
#         dataset, _ = self.data.load_it(
#             split=test_dataset_name, label_encoding=self.label_encoding
#         )  # The tfrecords dataset is batched

#         self.model.evaluate(dataset)

#         results = []
#         for files, labels in dataset:
#             predictions = self.model.predict(files, verbose=0)
#             for prediction, label in zip(predictions, labels.numpy()):
#                 results.append({"prediction": prediction, "actual": label})
#         self._calculate_confusion_matrix(results, test_dataset_name)
#         return self._calculate_accuracy(results)

#     def _calculate_accuracy(self, results):
#         correct = 0
#         total = 0
#         for result in results:
#             predicted_label = np.argmax(result["prediction"])
#             actual_label = (
#                 result["actual"]
#                 if self.label_encoding == "integer"
#                 else np.argmax(result["actual"])
#             )

#             if predicted_label == actual_label:
#                 correct += 1
#             total += 1
#         accuracy = correct / len(results)
#         print(f"Accuracy: {accuracy}")
#         return accuracy

#     def _calculate_confusion_matrix(self, results, dataset_name: str):
#         for result in results:
#             predicted_label = self.label_map.get(np.argmax(result["prediction"]))
#             if predicted_label:
#                 self.confusion_matrix.at[predicted_label, dataset_name] += 1
