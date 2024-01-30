
import logging
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')



class Label():
    def __init__(self, wav_id, relative_start_in_seconds, relative_end_in_seconds, label):
        self.wav_id = wav_id
        self.relative_start_in_seconds = relative_start_in_seconds
        self.relative_end_in_seconds = relative_end_in_seconds
        self.duration_in_seconds = relative_end_in_seconds - relative_start_in_seconds
        self.label = label
        # -----
        self.class_ = None
        self.split = None
        
    def map_label_to_class(self, label_to_class_map: Dict):
        for class_, labels in label_to_class_map.items():
            if self.label in labels:
                self.class_ = class_
                return
        raise ValueError(f"Could not map label {self.label} to a class")
    
    def set_split(self, split):
        self.split = split

    def get_hash(self):
        return f"{self.wav_id}_{self.relative_start_in_seconds}_{self.relative_end_in_seconds}_{self.label}"