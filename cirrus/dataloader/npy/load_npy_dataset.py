import numpy as np
import os
import pandas as pd
import tensorflow as tf
from typing import List, Dict
from sklearn.utils.class_weight import compute_class_weight
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

def label_encoder(labels: List[str]):
    unique_labels = np.unique(labels)
    unique_labels.sort()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_int[label] for label in labels])
    return encoded_labels, label_to_int

def calculate_class_weights(encoded_labels: np.ndarray):
    unique_labels = np.unique(encoded_labels)
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=encoded_labels)
    class_weights = dict(zip(range(len(unique_labels)), weights))
    return class_weights

def load_npy_dataset(df: pd.DataFrame, npy_path: str, batch_size: int = 32, shuffle: bool=True):
    logging.info("Loading %s dataset", df['split'].iloc[0])
    features_list = []
    labels_list = []
    for _, row in df.iterrows():
        npy_file_path = os.path.join(npy_path, f"{row['hash']}.npy")
        feature = np.load(npy_file_path)
        features_list.append(feature)
        labels_list.append(row.get('class'))
    features_array = np.array(features_list)
    encoded_labels, label_to_int_mapping = label_encoder(labels_list)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, encoded_labels))

    number_of_files = sum(1 for _ in dataset)
    logging.info("Found %s files belonging to %s classes", number_of_files, len(label_to_int_mapping))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    class_weights = calculate_class_weights(encoded_labels)
    return dataset, label_to_int_mapping, class_weights

