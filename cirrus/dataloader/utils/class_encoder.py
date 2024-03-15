from typing import List, Dict
import numpy as np
import json

import tensorflow as tf


class ClassEncoder:
    def __init__(self, label_class_map: Dict[str, List[str]]):
        self.label_class_map = label_class_map
        self.encoding_map = self.create_encoding_map()
        self.print_encoding_map()

    def create_encoding_map(self) -> Dict[str, Dict[str, List]]:
        encoding_map = {}
        classes = list(self.label_class_map.keys())
        for index, class_name in enumerate(classes):
            onehot_encoding = [0] * len(classes)
            onehot_encoding[index] = 1
            encoding_map[class_name] = {
                "integer_encoding": index,
                "onehot_encoding": onehot_encoding,
                "labels_belonging_to_class": self.label_class_map[class_name],
            }
        return encoding_map

    def encode_classes(self, classes: List[str], format: str = "one_hot"):
        encoded_classes = []
        for class_name in classes:
            if class_name in self.encoding_map:
                if format == "one_hot":
                    encoded_classes.append(
                        self.encoding_map[class_name]["onehot_encoding"]
                    )
                elif format == "integer":
                    encoded_classes.append(
                        self.encoding_map[class_name]["integer_encoding"]
                    )
            else:
                print(f"Class {class_name} not found in encoding map.")
        return np.array(encoded_classes)

    def print_encoding_map(self):
        print(json.dumps(self.encoding_map, indent=4))

    def decode_class(self, encoded_class, format: str = "one_hot"):
        if format == "integer" and len(self.encoding_map) > 2:
            raise ValueError("Cannot decode integer encoding for more than 2 classes.")

        if format == "one_hot":
            return self._decode_one_hot_class(encoded_class)
        elif format == "integer":
            return self._decode_integer_class(encoded_class)

    def _decode_one_hot_class(self, encoded_class):
        max_index = np.argmax(encoded_class)
        # Decode this index to the corresponding class name
        for class_name, info in self.encoding_map.items():
            if info["integer_encoding"] == max_index:
                return class_name

    def _decode_integer_class(self, encoded_class):
        # Transform to float if necessary
        if isinstance(encoded_class, np.ndarray):
            encoded_class = encoded_class[0]
        if isinstance(encoded_class, tf.Tensor):
            encoded_class = encoded_class.numpy()
        # Transform to integer
        encoded_class = 0 if encoded_class < 0.5 else 1
        # Decode this index to the corresponding class name
        for class_name, info in self.encoding_map.items():
            if encoded_class == info["integer_encoding"]:
                return class_name
        raise ValueError("Could not decode class.")


# def label_encoder(labels: List, format: str, classes: np.ndarray) -> np.ndarray:
#     """
#     Encode labels to a specific format
#     Args:
#         labels: List of labels
#         format: Format to encode the labels to. Supported formats: 'integer', 'one_hot'
#         classes: List of the different classes to ensure the encoding is consistent across different splits
#     Returns:
#         Encoded labels
#     """

#     assert len(labels) > 0, "Labels cannot be empty"
#     assert format in ["integer", "one_hot"], "Label format not supported"

#     # Update to use classes for consistent encoding
#     int_encoded_labels = _encode_to_integer(labels, classes)

#     if format == "integer":
#         return np.array(int_encoded_labels)
#     elif format == "one_hot":
#         return _encode_to_one_hot(int_encoded_labels, len(classes))


# def _encode_to_integer(labels: List, classes: np.ndarray):
#     label_to_int = {label: i for i, label in enumerate(classes)}
#     encoded_labels = [label_to_int[label] for label in labels if label in label_to_int]
#     return encoded_labels


# def _encode_to_one_hot(encoded_labels: List[int], num_classes: int):
#     one_hot_labels = np.eye(num_classes)[encoded_labels]
#     return one_hot_labels