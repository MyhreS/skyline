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

    def load(self, split, label_encoding):
        dataset_df = self._read_dataset_csv()
        assert split in dataset_df["split"].unique(), f"Split {split} does not exist"
        file_type = self._get_file_extension(dataset_df["hash"].iloc[0])
        classes = dataset_df["class"].unique()

        if split == "test":
            df_split = dataset_df[dataset_df["split"].str.contains("test")]
        else:
            df_split = dataset_df[dataset_df["split"] == split]
        if file_type == "tfrecord":
            return load_tfrecord_dataset(
                df_split, self.data_path, label_encoding, classes
            )
        elif file_type == "npy":
            return load_npy_dataset(df_split, self.data_path, label_encoding, classes)

    def get_names_of_test_datasets(self):
        df = self._read_dataset_csv()
        only_test_df = df[df["split"].str.contains("test")]
        return only_test_df["split"].unique().tolist()
