from .df_build.window import window
from .df_build.train_val_test_split import train_val_test_split
from .df_build.map_label_to_class import map_label_to_class
from .df_build.augment import augment
from .df_build.hash import hash
from .df_build.limit import limit
from .perform.perform import perform

import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

class Pipeline():
    def __init__(self, data_input_path: str, data_output_path: str):
        self.data_input_path = data_input_path
        self.data_output_path = data_output_path

        self.window_size = 1
        self.label_to_class_map = None
        self.augmentations = None
        self.audio_format = None
        self.sample_rate = None
        self.split = None
        self.file_type = None
        self.limit = None

    def _build(self, df: pd.DataFrame):
        window_size = self.window_size if self.window_size is not None else 1
        df = window(df, window_size)
        if self.split is not None:
            df = train_val_test_split(df, self.split['train'], self.split['test'], self.split['validation'])
        if self.label_to_class_map is not None:
            df = map_label_to_class(df, self.label_to_class_map)
        if self.sample_rate is not None:
            df['sample_rate'] = self.sample_rate
        if self.augmentations is not None:
            df = augment(df, self.augmentations)
        if self.limit is not None:
            df = limit(df, self.limit)
        if self.audio_format is not None:
            df['audio_format'] = self.audio_format
        if self.file_type is not None:
            df['file_type'] = self.file_type
        else:
            df['file_type'] = 'tfrecord'
        df = hash(df)
        return df
    
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
        

    def make(self, df: pd.DataFrame, clean=False):
        logging.info("Running pipeline")
        build_df = self._build(df)
        perform(build_df, self.data_input_path, self.data_output_path, clean=clean)


    

        

