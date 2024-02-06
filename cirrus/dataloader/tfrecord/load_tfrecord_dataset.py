import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')


def _parse_function(proto):
    # Include 'shape' in the feature description
    feature_description = {
        'spectrogram': tf.io.VarLenFeature(tf.float32),
        'shape': tf.io.FixedLenFeature([2], tf.int64),  # Adjust [2] based on the dimensions of your spectrogram
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    spectrogram = tf.sparse.to_dense(parsed_features['spectrogram'])
    shape = parsed_features['shape']
    
    spectrogram = tf.reshape(spectrogram, shape)
    return spectrogram


def attach_labels(spectrogram, label):
    return spectrogram, label

def create_dataset_from_tfrecords(tfrecord_file_paths, labels):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
    # Convert labels to a Tensor to use them within the map function
    labels_tensor = tf.constant(labels, dtype=tf.int64)
    # Map _parse_function to decode spectrograms and attach labels using the indices
    spectrogram_dataset = raw_dataset.enumerate().map(lambda idx, proto: attach_labels(_parse_function(proto), labels_tensor[idx]))
    return spectrogram_dataset

def label_encoder(labels):
    unique_labels = np.unique(labels)
    unique_labels.sort()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = [label_to_int[label] for label in labels]
    return encoded_labels, label_to_int

def calculate_class_weights(encoded_labels):
    unique_labels = np.unique(encoded_labels)
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=encoded_labels)
    class_weights = dict(zip(range(len(unique_labels)), weights))
    return class_weights
    



def load_tfrecord_dataset(df: pd.DataFrame, tfrecord_path: str, batch_size: int = 32, shuffle=True):
    logging.info("Loading %s dataset", df['split'].iloc[0])
    
    labels = df['class'].tolist() 
    encoded_labels, label_to_int_mapping = label_encoder(labels)
    
    tfrecord_file_paths = [os.path.join(tfrecord_path, f"{hash_}.tfrecord") for hash_ in df['hash']]
    tfrecords_dataset = create_dataset_from_tfrecords(tfrecord_file_paths, encoded_labels)

    number_of_files = sum(1 for _ in tfrecords_dataset)
    logging.info("Found %s files belonging to %s classes", number_of_files, len(label_to_int_mapping))
    if shuffle:
        tfrecords_dataset = tfrecords_dataset.shuffle(buffer_size=1000)
    tfrecords_dataset = tfrecords_dataset.batch(batch_size)

    class_weights = calculate_class_weights(encoded_labels)
    
    return tfrecords_dataset, label_to_int_mapping, class_weights










