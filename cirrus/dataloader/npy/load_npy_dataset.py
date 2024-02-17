import numpy as np
import os
import pandas as pd
import tensorflow as tf
from ..utils.label_encoder import label_encoder
from ..utils.class_weight_calculator import class_weight_calculator
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def load_npy_dataset(
    df: pd.DataFrame,
    npy_path: str,
    label_encoding: str,
    classes: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
):
    logging.info("Loading %s dataset", df["split"].iloc[0])
    features_list = []
    labels_list = []
    for _, row in df.iterrows():
        npy_file_path = os.path.join(npy_path, f"{row['hash']}.npy")
        feature = np.load(npy_file_path)
        features_list.append(feature)
        labels_list.append(row.get("class"))
    features_array = np.array(features_list)
    encoded_labels = label_encoder(labels_list, label_encoding, classes)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, encoded_labels))

    number_of_files = sum(1 for _ in dataset)
    number_of_unique_labels = len(df["class"].unique())
    logging.info(
        "Found %s files belonging to %s classes",
        number_of_files,
        len(number_of_unique_labels),
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)

    shape = features_array[0].shape

    if df["split"].iloc[0] == "train":
        class_weights = class_weight_calculator(encoded_labels)
        return dataset, shape, class_weights
    return dataset, shape
