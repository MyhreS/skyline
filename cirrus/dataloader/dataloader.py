import json
import pandas as pd
import os
from .utils.load_tfrecord_dataset import load_tfrecord_dataset
from typing import List
from .utils.class_encoder import ClassEncoder

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


class Dataloader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataset_df = []
        self.label_encoder = None

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

    def _verify_label_mapping_is_similar(self, label_maps: List):
        previous_label_map = None
        for label_map in label_maps:
            if previous_label_map is not None:
                assert (
                    previous_label_map == label_map
                ), "The label mapping must be the same for all splits"
            previous_label_map = label_map

    def get_dataset_df(self):
        if not self.dataset_df:
            self.dataset_df = self._read_dataset_csv()
        return self.dataset_df

    def get_class_encoder(self):
        # Read json file
        if not self.label_encoder:
            with open("cache/label_class_map.json") as f:
                label_class_map = json.load(f)
            self.label_encoder = ClassEncoder(label_class_map)
        return self.label_encoder

    def load(self, split, label_encoding):
        dataset_df = self.get_dataset_df()
        class_encoder = self.get_class_encoder()

        if split == "test":
            df_split = dataset_df[dataset_df["split"].str.contains("test")]
        else:
            df_split = dataset_df[dataset_df["split"] == split]
            
        return load_tfrecord_dataset(
            split, df_split, self.data_path, label_encoding, class_encoder
        )

    def get_names_of_test_datasets(self):
        df = self._read_dataset_csv()
        only_test_df = df[df["split"].str.contains("test")]
        return only_test_df["split"].unique().tolist()
