import pandas as pd
import os
import numpy as np
import tensorflow as tf
from .class_encoder import ClassEncoder
from .class_weight_calculator import class_weight_calculator

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def _parse_function(proto):
    feature_description = {
        "audio": tf.io.VarLenFeature(tf.float32),
        "shape": tf.io.VarLenFeature(tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    audio = tf.sparse.to_dense(parsed_features["audio"])
    shape = tf.sparse.to_dense(parsed_features["shape"])

    audio = tf.reshape(audio, tf.cast(shape, tf.int32))
    audio = tf.expand_dims(audio, -1)
    return audio


def attach_labels(audio, label):
    return audio, label


def create_dataset_from_tfrecords(tfrecord_file_paths, labels):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
    labels_tensor = tf.constant(labels, dtype=tf.int64)
    audio_dataset = raw_dataset.enumerate().map(
        lambda idx, proto: attach_labels(_parse_function(proto), labels_tensor[idx])
    )
    return audio_dataset


def load_tfrecord_dataset(
    name: str,
    df: pd.DataFrame,
    tfrecord_path: str,
    label_encoding: str,
    class_encoder: ClassEncoder,
    batch_size: int = 32,
    shuffle=True,
):
    logging.info("Loading %s dataset", name)

    labels = df["class"].tolist()
    encoded_labels = class_encoder.encode_classes(labels, label_encoding)

    tfrecord_file_paths = [
        os.path.join(tfrecord_path, f"{hash_}.tfrecord") for hash_ in df["hash"]
    ]
    tfrecords_dataset = create_dataset_from_tfrecords(
        tfrecord_file_paths, encoded_labels
    )
    dataset_unique_labels = len(df["class"].unique())
    number_of_files = sum(1 for _ in tfrecords_dataset)
    logging.info(
        "Found %s files belonging to %s classes", number_of_files, dataset_unique_labels
    )
    if not name.startswith("test"):
        if shuffle:
            tfrecords_dataset = tfrecords_dataset.shuffle(buffer_size=1000)

    tfrecords_dataset = tfrecords_dataset.batch(batch_size)

    for features, labels in tfrecords_dataset.take(1):
        shape = features[0].shape

    if df.iloc[0]["split"] == "train":
        class_weights = class_weight_calculator(encoded_labels)
        return tfrecords_dataset, shape, class_weights
    return tfrecords_dataset, shape
