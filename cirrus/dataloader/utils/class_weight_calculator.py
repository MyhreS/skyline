from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from typing import Dict

def class_weight_calculator(encoded_labels: np.ndarray):
    unique_labels = np.unique(encoded_labels)
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=encoded_labels)
    class_weights = dict(zip(range(len(unique_labels)), weights))
    return class_weights