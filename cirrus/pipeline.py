from .utils.window_dataframe_wavs import window_dataframe_wavs
from .utils.split_dataframe_wavs import split_dataframe_wavs
from .utils.map_dataframe_label_to_class import map_dataframe_label_to_class
from .utils.augment_dataframe_wavs import augment_dataframe_wavs
from .utils.hash_dataframe_wavs import hash_dataframe_wavs
from .utils.write_dataframe_wavs import write_dataframe_wavs

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
        df = self._hash(df)
        return df
    
    def _hash(self, df):
        df = self._file_type(df)
        hashed_df = hash_dataframe_wavs(df)
        return hashed_df
    
    def _file_type(self, df):
        df = self._audio_format_audio(df)
        if self.file_type is None:
            df['file_type'] = 'npy'
        else:
            df['file_type'] = self.file_type
        return df

    def _audio_format_audio(self, df):
        df = self._augment(df)
        if self.audio_format is None:
            return df
        df['audio_format'] = self.audio_format
        return df
    
    def _augment(self, df):
        df = self._sample_rate(df)
        if self.augmentations is None:
            return df
        augment_df = augment_dataframe_wavs(df, self.augmentations)
        return augment_df

    def _sample_rate(self, df):
        df = self._label_to_class(df)
        if self.sample_rate is None:
            return df
        df['sample_rate'] = self.sample_rate
        return df
    
    
    def _label_to_class(self, df):
        df = self._split(df)
        if self.label_to_class_map is None:
            return df
        label_to_class_df = map_dataframe_label_to_class(df, self.label_to_class_map)
        return label_to_class_df

    def _split(self, df):
        # TODO: Make it also allow non uniform label_duration_sec
        # TODO: Now it splits on label, make it split on class as an option
        df = self._window(df)
        if self.split is None:
            return df

        split_df = split_dataframe_wavs(df, self.split['train'], self.split['test'], self.split['validation'])
        return split_df
    
    def _window(self, df): # Note: Only supports integer seconds
        if self.window_size is None:
            self.window_size = 1
        windowed_df = window_dataframe_wavs(df, self.window_size)
        return windowed_df
    

    # All methods for performing the recipe
    def _perform(self, df: pd.DataFrame, input_path_to_data: str, output_path_to_data: str):
        write_dataframe_wavs(df, input_path_to_data, output_path_to_data)

