import pandas as pd
import os
from .tfrecord.load_tfrecord_dataset import load_tfrecord_dataset
from .npy.load_npy_dataset import load_npy_dataset

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

class Dataloader():
    def __init__(self, data_path: str):
        self.data_path = data_path

    def _read_dataset_csv(self):
        dataset_df = pd.read_csv("cache/dataset.csv")
        assert 'hash' in dataset_df.columns, "The dataset.csv file must contain a column called 'hash'"
        assert 'split' in dataset_df.columns, "The dataset.csv file must contain a column called 'split'"
        assert 'class' in dataset_df.columns, "The dataset.csv file must contain a column called 'class'"
        assert len(dataset_df) > 0, "The dataset.csv file must contain at least one row"
        return dataset_df
    

    def _get_file_extension(self, file_name: str):
        possible_extensions = ["tfrecord", "npy"]
        for possible_extension in possible_extensions:
            file_path = os.path.join(self.data_path, f"{file_name}.{possible_extension}")
            if os.path.exists(file_path):
                return possible_extension
        raise ValueError(f"Could not find any fitting dataloaders")

    def load(self):
        dataset_df = self._read_dataset_csv()
        file_type = self._get_file_extension(dataset_df['hash'].iloc[0])
        dataset_df_train = dataset_df[dataset_df['split'] == 'train']
        dataset_df_val = dataset_df[dataset_df['split'] == 'validation']
        dataset_df_test = dataset_df[dataset_df['split'] == 'test']

        if file_type == "tfrecord":
            train_tfrecords_dataset, label_to_int_mapping, class_weights, shape = load_tfrecord_dataset(dataset_df_train, self.data_path)
            val_tfrecords_dataset, _, _, _ = load_tfrecord_dataset(dataset_df_val, self.data_path)
            test_tfrecords_dataset, _, _, _ = load_tfrecord_dataset(dataset_df_test, self.data_path)
            return train_tfrecords_dataset, val_tfrecords_dataset, test_tfrecords_dataset, label_to_int_mapping, class_weights, shape
        elif file_type == "npy":
            train_npy_dataset, label_to_int_mapping, class_weights, shape = load_npy_dataset(dataset_df_train, self.data_path)
            val_npy_dataset, _, _, _= load_npy_dataset(dataset_df_val, self.data_path)
            test_npy_dataset, _, _, _= load_npy_dataset(dataset_df_test, self.data_path)
            return train_npy_dataset, val_npy_dataset, test_npy_dataset, label_to_int_mapping, class_weights, shape
    
