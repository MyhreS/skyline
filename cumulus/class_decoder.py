from typing import List, Dict
import numpy as np
import json

import tensorflow as tf


class ClassDecoder:
    def __init__(self, label_class_map: Dict[str, List[str]]):
        self.label_class_map = label_class_map
        self.encoding_map = self.create_encoding_map()
        self.print_encoding_map()

    def create_encoding_map(self) -> Dict[str, Dict[str, List]]:
        encoding_map = {}
        classes = list(self.label_class_map.keys())
        sorted_classes_alphabetically = sorted(classes)
        for index, class_name in enumerate(sorted_classes_alphabetically):
            encoding_map[class_name] = {
                "integer_encoding": index,
                "labels_belonging_to_class": self.label_class_map[class_name],
            }
        return encoding_map

    def print_encoding_map(self):
        print(json.dumps(self.encoding_map, indent=4))

    def decode(self, encoded_label):
        if (
            type(encoded_label) is int
            or type(encoded_label) is float
            or type(encoded_label) is np.int32
        ):
            label_int = encoded_label
        else:
            assert (
                type(encoded_label) == np.ndarray
            ), "Encoded label must be a numpy array."

            if len(encoded_label.shape) > 1:
                label_int = np.argmax(encoded_label)
            else:
                label_int = 0 if encoded_label <= 0.5 else 1

        for class_name, info in self.encoding_map.items():
            # Check if it matches integer encoding
            if info["integer_encoding"] == label_int:
                return class_name

        raise ValueError("Could not decode label.")

    # def decode_class(self, encoded_class, format: str = "one_hot"):
    #     if format == "integer" and len(self.encoding_map) > 2:
    #         raise ValueError("Cannot decode integer encoding for more than 2 classes.")

    #     if format == "one_hot":
    #         return self._decode_one_hot_class(encoded_class)
    #     elif format == "integer":
    #         return self._decode_integer_class(encoded_class)

    # def _decode_one_hot_class(self, encoded_class):
    #     max_index = np.argmax(encoded_class)
    #     # Decode this index to the corresponding class name
    #     for class_name, info in self.encoding_map.items():
    #         if info["integer_encoding"] == max_index:
    #             return class_name

    # def _decode_integer_class(self, encoded_class):
    #     # Transform to float if necessary
    #     if isinstance(encoded_class, np.ndarray):
    #         encoded_class = encoded_class[0]
    #     if isinstance(encoded_class, tf.Tensor):
    #         encoded_class = encoded_class.numpy()
    #     # Transform to integer
    #     encoded_class = 0 if encoded_class < 0.5 else 1
    #     # Decode this index to the corresponding class name
    #     for class_name, info in self.encoding_map.items():
    #         if encoded_class == info["integer_encoding"]:
    #             return class_name
    #     raise ValueError("Could not decode class.")
