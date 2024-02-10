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
    encoded_labels, label_to_int_mapping = label_encoder(labels_list, label_encoding)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, encoded_labels))

    number_of_files = sum(1 for _ in dataset)
    logging.info(
        "Found %s files belonging to %s classes",
        number_of_files,
        len(label_to_int_mapping),
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    class_weights = class_weight_calculator(encoded_labels)

    shape = features_array[0].shape

    return dataset, label_to_int_mapping, class_weights, shape
