import pandas as pd
from .tfrecord.load_tfrecord_dataset import load_tfrecord_dataset

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

class Dataloader():
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataset_df = None

    def _read_dataset_csv(self):
        self.dataset_df = pd.read_csv("cache/dataset.csv")

    def load_training_dataset(self):
        if self.dataset_df is None:
            self._read_dataset_csv()
        dataset_df_train = self.dataset_df[self.dataset_df['split'] == 'train']
        train_dataset = load_tfrecord_dataset(dataset_df_train, self.data_path)


    def load_validation_dataset(self):
        pass

    def load_test_dataset(self):
        pass
