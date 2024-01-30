
import os
import shutil
import pandas as pd
import os
from typing import Dict, List, T

from .pipeline import Pipeline
from .augmenter import Augmenter

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')


class Data():
    """
    A class that handles everything regarding the data. It loads the data, does all processing with the data, writes the data and describes the data.
    """

    def __init__(self, input_path_to_data):
        self.input_path_to_data = input_path_to_data
        self.metadata_df = self._get_metadata_df()
        self._check_wavs()
        self.pipeline = Pipeline()

    def _get_metadata_df(self):
        path_to_metadata = os.path.join(self.input_path_to_data, "data.csv")
        self._validate_metadata_exists(path_to_metadata)
        metadata_df = pd.read_csv(path_to_metadata)
        self._validate_metadata_df(metadata_df)
        return metadata_df

    def _validate_metadata_exists(self, path_to_metadata):
        if not os.path.exists(path_to_metadata):
            raise ValueError(f"Could not find metadata file at {path_to_metadata}")

    def _validate_metadata_df(self, metadata_df):
        # Should check that the metadata_df has the colums: wav_blob, label, relative_start_sec, relative_end_sec, duration_sec
        self._validate_metadata_df_columns(metadata_df)
        self._validate_metadata_df_size(metadata_df)
        
    def _validate_metadata_df_columns(self, metadata_df):
        should_contain_colums = ['wav_blob', 'wav_duration_sec', 'label_duration_sec', 'label_relative_start_sec', 'label_relative_end_sec']
        for column in should_contain_colums:
            if column not in metadata_df.columns:
                raise ValueError(f"Metadata df does not contain column {column}")
    
    def _validate_metadata_df_size(self, metadata_df):
        if metadata_df.shape[0] == 0:
            raise ValueError("Metadata df has no rows")
        
    def _check_wavs(self):
        path_to_wavs_folder = os.path.join(self.input_path_to_data, "wavs")
        self._validate_wavs_folder_exists(path_to_wavs_folder)
        wavs_names = os.listdir(path_to_wavs_folder)
        self._validate_wavs_exists(path_to_wavs_folder, wavs_names)
        return 

    
    def _validate_wavs_folder_exists(self, path_to_wavs_folder):
        if not os.path.exists(path_to_wavs_folder):
            raise ValueError(f"Could not find wavs folder at {path_to_wavs_folder}")
    
    def _validate_wavs_exists(self, path_to_wavs_folder, wavs_names):
        for wav_name in wavs_names:
            path_to_wav = os.path.join(path_to_wavs_folder, wav_name)
            if not os.path.exists(path_to_wav):
                raise ValueError(f"Could not find wav at {path_to_wav}")

    # List of functions which builds the pipeline / recipe for the data    
    def window_it(self, window_size_in_seconds: int):
        """
        Set the window size for the data
        """
        self.pipeline.window_size = window_size_in_seconds

    def label_to_class_map_it(self, label_to_class_map: Dict):
        """
        Set the mapping from label to class for the data
        """
        self.pipeline.label_to_class_map = label_to_class_map

    def augment_it(self, augmentations: List):
        """
        Set the augmentation steps for the data
        """
        for augmentation in augmentations:
            if augmentation not in Augmenter.augment_options:
                raise ValueError(f"Augmentation {augmentation} not in possible augmentations {Augmenter.augment_options}")
        self.pipeline.augmentations = augmentations

    def audio_format_it(self, audio_format: str):
        """
        Set the audio format for the data
        """
        possible_audio_formats = ["stft", 'log_mel']
        if audio_format not in possible_audio_formats:
            raise ValueError(f"Audio format {audio_format} not in possible audio formats {possible_audio_formats}")
        self.pipeline.audio_format = audio_format

    def sample_rate_it(self, sample_rate: int):
        """
        Set the sample rate for the data
        """
        self.pipeline.sample_rate = sample_rate

    def split_it(self, train_percent: int, test_percent: int, validation_percent: int):
        """
        Set the split for the data
        """
        if train_percent + test_percent + validation_percent != 100:
            raise ValueError('Split percentages must add up to 100')
        self.pipeline.split = {
            'train': train_percent,
            'test': test_percent,
            'validation': validation_percent
        }

    def file_type_it(self, file_type: str):
        """
        Set the file type for the data
        """
        if file_type not in ['npy']:
            raise ValueError(f"Invalid file_type {file_type}. Allowed file_types: npy")
        self.pipeline.file_type = file_type


    # List of functions to describe or perform the pipeline / recipe for the data
    def describe_it(self):
        """
        Describe the data / pipeline / recipe for the data
        """
        self.pipeline.describe(self.metadata_df)

    def run_it(self, output_path_to_data: str):
        """
        Run the pipeline / recipe for the data
        """
        self.pipeline.run(self.metadata_df, self.input_path_to_data, output_path_to_data)
