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
        results = {}
        # Test the entire dataset
        logging.info("Average accuracy")
        test_dataset, *_ = self.data.load_it(
            split="test", label_encoding=self.label_encoding
        )
        result = self.model.evaluate(test_dataset)
        results["average"] = result

        # Test the sub datasets
        for test_dataset in self.data.dataloader.get_names_of_test_datasets():
            logging.info(f"Accuracy on {test_dataset}")
            dataset, _ = self.data.load_it(
                split=test_dataset, label_encoding=self.label_encoding
            )
            result = self.model.evaluate(dataset)
            results[test_dataset] = result

        self.logger.save_model_test_accuracy(results)
