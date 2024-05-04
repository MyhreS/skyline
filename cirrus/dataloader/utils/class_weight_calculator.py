from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from typing import Dict

"""
Function for calculating class weights for balancing the classes in training.
"""


def class_weight_calculator(encoded_labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for balancing the classes in training.
    Works with both integer encoded and one-hot encoded labels.

    Args:
        encoded_labels: np.ndarray - The encoded labels, either integer or one-hot encoded.

    Returns:
        class_weights: Dict[int, float] - A dictionary mapping class indices to their weights.
    """
    if encoded_labels.ndim > 1 and np.unique(encoded_labels).size == 2:
        labels_int = np.argmax(encoded_labels, axis=1)
    else:
        labels_int = encoded_labels
    unique_labels = np.unique(labels_int)
    weights = compute_class_weight(
        class_weight="balanced", classes=unique_labels, y=labels_int
    )
    class_weights = dict(zip(unique_labels, weights))
    return class_weights
