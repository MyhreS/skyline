from typing import List, Dict, Union
import numpy as np

def label_encoder(labels: List, format: str) -> Union[List, np.ndarray, Dict]:
    """
    Encode labels to a specific format
    Args:
    labels: List of labels
    format: Format to encode the labels to. Supported formats: 'integer', 'one_hot', 'binary'
    Returns:
    Encoded labels and label to integer mapping
    """

    assert len(labels) > 0, "Labels cannot be empty"
    assert format in ['integer', 'one_hot', 'binary'], "Label format not supported"
    if format == 'integer':
        return _encode_to_integer(labels)
    elif format == 'one_hot':
        return _encode_to_one_hot(labels)
    elif format == 'binary':
        return _encode_to_binary(labels)

def _encode_to_integer(labels: List):
    unique_labels = np.unique(labels)
    unique_labels.sort()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = [label_to_int[label] for label in labels]
    return encoded_labels, label_to_int

def _encode_to_one_hot(labels: List):
    unique_labels = np.unique(labels)
    unique_labels.sort()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    one_hot_encoded = np.zeros((len(labels), len(unique_labels)))
    for i, label in enumerate(labels):
        one_hot_encoded[i, label_to_int[label]] = 1
    return one_hot_encoded, label_to_int

def _encode_to_binary(labels: List):
    unique_labels = np.unique(labels)
    unique_labels.sort()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    num_bits = int(np.ceil(np.log2(len(unique_labels))))
    binary_encoded = np.zeros((len(labels), num_bits))
    for i, label in enumerate(labels):
        binary_code = [int(x) for x in bin(label_to_int[label])[2:].zfill(num_bits)]
        binary_encoded[i] = binary_code
    return binary_encoded, label_to_int
