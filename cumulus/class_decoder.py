from typing import List, Dict
import numpy as np
import json

import tensorflow as tf


def to_int(value):
    if (
        type(value) is int
        or type(value) is float
        or type(value) is np.int32
        or type(value) is np.int64
    ):
        value_int = value
    else:
        assert type(value) == np.ndarray, "Encoded label must be a numpy array."
        if len(value) > 1:
            value_int = np.argmax(value)
        else:
            value_int = 0 if value <= 0.5 else 1
    return value_int


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
        label_int = to_int(encoded_label)
        for class_name, info in self.encoding_map.items():
            if info["integer_encoding"] == label_int:
                return class_name

        raise ValueError("Could not decode label.")
    
    

