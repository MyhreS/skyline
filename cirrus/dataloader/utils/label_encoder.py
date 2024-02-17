from typing import List
import numpy as np


def label_encoder(labels: List, format: str, classes: np.ndarray) -> np.ndarray:
    """
    Encode labels to a specific format
    Args:
        labels: List of labels
        format: Format to encode the labels to. Supported formats: 'integer', 'one_hot'
        classes: List of the different classes to ensure the encoding is consistent across different splits
    Returns:
        Encoded labels
    """

    assert len(labels) > 0, "Labels cannot be empty"
    assert format in ["integer", "one_hot"], "Label format not supported"

    # Update to use classes for consistent encoding
    int_encoded_labels = _encode_to_integer(labels, classes)

    if format == "integer":
        return np.array(int_encoded_labels)
    elif format == "one_hot":
        return _encode_to_one_hot(int_encoded_labels, len(classes))


def _encode_to_integer(labels: List, classes: np.ndarray):
    label_to_int = {label: i for i, label in enumerate(classes)}
    encoded_labels = [label_to_int[label] for label in labels if label in label_to_int]
    return encoded_labels


def _encode_to_one_hot(encoded_labels: List[int], num_classes: int):
    one_hot_labels = np.eye(num_classes)[encoded_labels]
    return one_hot_labels
