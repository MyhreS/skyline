import pandas as pd
import os
import numpy as np
import tensorflow as tf
from ..utils.label_encoder import label_encoder
from ..utils.class_weight_calculator import class_weight_calculator

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def _parse_function(proto):
    # Adjust 'shape' in the feature description to handle variable dimensions
    feature_description = {
        "audio": tf.io.VarLenFeature(tf.float32),
        "shape": tf.io.VarLenFeature(tf.int64),  # Use VarLenFeature for variable length
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    audio = tf.sparse.to_dense(parsed_features["audio"])
    shape = tf.sparse.to_dense(parsed_features["shape"])

    audio = tf.reshape(audio, tf.cast(shape, tf.int32))  # Ensure shape is cast to tf.int32 for tf.reshape
    return audio


def attach_labels(audio, label):
    return audio, label


def create_dataset_from_tfrecords(tfrecord_file_paths, labels):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
    # Convert labels to a Tensor to use them within the map function
    labels_tensor = tf.constant(labels, dtype=tf.int64)
    # Map _parse_function to decode audio and attach labels using the indices
    audio_dataset = raw_dataset.enumerate().map(
        lambda idx, proto: attach_labels(_parse_function(proto), labels_tensor[idx])
    )
    return audio_dataset


def load_tfrecord_dataset(
    df: pd.DataFrame,
    tfrecord_path: str,
    label_encoding: str,
    batch_size: int = 32,
    shuffle=True,
):
    logging.info("Loading %s dataset", df["split"].iloc[0])

    labels = df["class"].tolist()
    encoded_labels, label_to_int_mapping = label_encoder(labels, label_encoding)

    tfrecord_file_paths = [
        os.path.join(tfrecord_path, f"{hash_}.tfrecord") for hash_ in df["hash"]
    ]
    tfrecords_dataset = create_dataset_from_tfrecords(
        tfrecord_file_paths, encoded_labels
    )

    number_of_files = sum(1 for _ in tfrecords_dataset)
    logging.info(
        "Found %s files belonging to %s classes",
        number_of_files,
        len(label_to_int_mapping),
    )
    if shuffle:
        tfrecords_dataset = tfrecords_dataset.shuffle(buffer_size=1000)
    tfrecords_dataset = tfrecords_dataset.batch(batch_size)

    class_weights = class_weight_calculator(encoded_labels)

    for features, labels in tfrecords_dataset.take(1):
        shape = features[0].shape

    return tfrecords_dataset, label_to_int_mapping, class_weights, shape
