import pandas as pd
import os
from .tfrecord.load_tfrecord_dataset import load_tfrecord_dataset
from .npy.load_npy_dataset import load_npy_dataset
from typing import List

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


class Dataloader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def _read_dataset_csv(self):
        dataset_df = pd.read_csv("cache/dataset.csv")
        assert (
            "hash" in dataset_df.columns
        ), "The dataset.csv file must contain a column called 'hash'"
        assert (
            "split" in dataset_df.columns
        ), "The dataset.csv file must contain a column called 'split'"
        assert (
            "class" in dataset_df.columns
        ), "The dataset.csv file must contain a column called 'class'"
        assert len(dataset_df) > 0, "The dataset.csv file must contain at least one row"
        return dataset_df

    def _get_file_extension(self, file_name: str):
        possible_extensions = ["tfrecord", "npy"]
        for possible_extension in possible_extensions:
            file_path = os.path.join(
                self.data_path, f"{file_name}.{possible_extension}"
            )
            if os.path.exists(file_path):
                return possible_extension
        raise ValueError(f"Could not find any fitting dataloaders")

    def _verify_label_mapping_is_similar(self, label_maps: List):
        previous_label_map = None
        for label_map in label_maps:
            if previous_label_map is not None:
                assert (
                    previous_label_map == label_map
                ), "The label mapping must be the same for all splits"
            previous_label_map = label_map

    def load(self, label_encoding):
        dataset_df = self._read_dataset_csv()
        file_type = self._get_file_extension(dataset_df["hash"].iloc[0])

        # Splitting the dataset based on its phase
        splits = {
            split: dataset_df[dataset_df["split"] == split]
            for split in ["train", "validation", "test"]
        }

        # Function to load dataset based on file type
        def load_dataset(df, data_path, label_encoding, file_type):
            if file_type == "tfrecord":
                return load_tfrecord_dataset(df, data_path, label_encoding)
            elif file_type == "npy":
                return load_npy_dataset(df, data_path, label_encoding)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        datasets = {}
        label_mappings = {}
        for split, df in splits.items():
            dataset, label_to_int_mapping, *rest = load_dataset(
                df, self.data_path, label_encoding, file_type
            )
            datasets[split] = dataset
            label_mappings[split] = label_to_int_mapping

        self._verify_label_mapping_is_similar(list(label_mappings.values()))

        # Assuming class_weights and shape are the same for all splits if applicable
        class_weights, shape = rest if rest else (None, None)

        return (
            datasets["train"],
            datasets["validation"],
            datasets["test"],
            label_mappings["train"],
            class_weights,
            shape,
        )
