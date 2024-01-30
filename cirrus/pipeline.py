from .build_dataframe.window import window
from .build_dataframe.train_val_test_split import train_val_test_split
from .build_dataframe.map_label_to_class import map_label_to_class
from .build_dataframe.augment import augment
from .build_dataframe.hash import hash
from .perform_pipeline.perform_pipeline import perform_pipeline
from .augmenter import Augmenter

import pandas as pd
from typing import List, Dict
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

class Pipeline():
    def __init__(self):
        self.window_size = None
        self.label_to_class_map = None
        self.augmentations = None
        self.audio_format = None
        self.sample_rate = None
        self.split = None
        self.file_type = None

    
    def describe(self, df: pd.DataFrame):
        logging.info("Describing data")
        logging.info("Original data shape %s", df.shape)
        logging.info("Original data wav duration sec %s", df.groupby('wav_blob')['wav_duration_sec'].first().sum())
        logging.info("Original data label duration sec %s", df['label_duration_sec'].sum())
        logging.info("Original data head(5)\n%s", df.head(5))

        df = self._build(df)
        logging.info("Resulting data shape %s", df.shape)
        logging.info("Resulting data wav duration sec %s", df.groupby('wav_blob')['wav_duration_sec'].first().sum())
        logging.info("Resulting data label duration sec %s", df['label_duration_sec'].sum())
        
        if 'split' in df.columns:
            for split in df['split'].unique():
                logging.info("--------")
                logging.info("- Resulting data for split %s:", split)
                logging.info("- Split shape %s", df[df['split'] == split].shape)
                if 'class' in df.columns:
                    for class_ in df['class'].unique():
                        logging.info("-- Class %s shape %s", class_, df[(df['split'] == split) & (df['class'] == class_)].shape)
                for label in df['label'].unique():
                    logging.info("-- Label %s shape %s", label, df[(df['split'] == split) & (df['label'] == label)].shape)  
        else:
            logging.info("--------")
            if 'class' in df.columns:
                for class_ in df['class'].unique():
                    logging.info("- Class %s shape %s", class_, df[df['class'] == class_].shape)
            for label in df['label'].unique():
                logging.info("- Label %s shape %s", label, df[df['label'] == label].shape)
        logging.info("Resulting data head(5)\n%s", df.head(5))
        

    def run(self, df: pd.DataFrame, input_path_to_data: str, output_path_to_data: str):
        logging.info("Running pipeline")
        build_df = self._build(df)
        self._perform(build_df, input_path_to_data, output_path_to_data)
    


    # Methods for building  the pipeline recipe. The recipe is a dataframe.
    def _build(self, df: pd.DataFrame):
        # Apply windowing
        window_size = self.window_size if self.window_size is not None else 1
        df = window(df, window_size)
        # Apply data splitting
        if self.split is not None:
            df = train_val_test_split(df, self.split['train'], self.split['test'], self.split['validation'])
        # Apply label to class mapping
        if self.label_to_class_map is not None:
            df = map_label_to_class(df, self.label_to_class_map)
        # Apply sample rate transformation
        if self.sample_rate is not None:
            df['sample_rate'] = self.sample_rate
        # Apply augmentations
        if self.augmentations is not None:
            df = augment(df, self.augmentations)
        # Apply audio format transformation
        if self.audio_format is not None:
            df['audio_format'] = self.audio_format
        # Apply file type transformation
        if self.file_type is not None:
            df['file_type'] = self.file_type
        else:
            df['file_type'] = 'npy'
        # Apply hashing
        df = hash(df)
        return df

    

    # All methods for performing the recipe
    def _perform(self, df: pd.DataFrame, input_path_to_data: str, output_path_to_data: str):
        perform_pipeline(df,input_path_to_data, output_path_to_data)

