import os
import pandas as pd
import os
from typing import Dict, List

from .preprocesser.preprocesser import Preprocesser
from .dataloader.dataloader import Dataloader
from .utils.augmenter.augmenter import Augmenter
from .utils.audio_formatter.audio_formatter import AudioFormatter
from .utils.file_typer.file_typer import FileTyper

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)

# TODO list:
# Make function for viewing the data, like a plot function to look at the spectogram or listen to the audio wav
# Make function like get_shape, get_class_weights, get_class_mapping etc
# Support raw waveforms


class Data:
    """
    A class that handles everything regarding the data. It loads the data, does all processing with the data, writes the data and describes the data.
    """

    def __init__(self, data_input_path, data_output_path):
        self.data_input_path = data_input_path
        self.data_output_path = data_output_path

        self.metadata_df = self._get_metadata_df()
        self._check_wavs()
        self.pipeline = Preprocesser(data_input_path, data_output_path)
        self.dataloader = Dataloader(data_output_path)

    def _get_metadata_df(self):
        path_to_metadata = os.path.join(self.data_input_path, "data.csv")
        self._validate_metadata_exists(path_to_metadata)
        metadata_df = pd.read_csv(path_to_metadata)
        self._validate_metadata_df(metadata_df)
        return metadata_df

    def _validate_metadata_exists(self, path_to_metadata):
        if not os.path.exists(path_to_metadata):
            raise ValueError(f"Could not find metadata file at {path_to_metadata}")

    def _validate_metadata_df(self, metadata_df):
        self._validate_metadata_df_columns(metadata_df)
        self._validate_metadata_df_size(metadata_df)

    def _validate_metadata_df_columns(self, metadata_df):
        should_contain_colums = [
            "file_name",
            "wav_duration_sec",
            "label",
            "label_duration_sec",
            "label_relative_start_sec",
            "label_relative_end_sec",
        ]
        for column in should_contain_colums:
            if column not in metadata_df.columns:
                raise ValueError(f"Metadata df does not contain column {column}")

    def _validate_metadata_df_size(self, metadata_df):
        if metadata_df.shape[0] == 0:
            raise ValueError("Metadata df has no rows")

    def _check_wavs(self):
        path_to_wavs_folder = os.path.join(self.data_input_path, "wavs")
        self._validate_wavs_folder_exists(path_to_wavs_folder)
        wavs_names = os.listdir(path_to_wavs_folder)
        self._filter_metadata_df_on_wav_names(wavs_names)
        return

    def _validate_wavs_folder_exists(self, path_to_wavs_folder):
        if not os.path.exists(path_to_wavs_folder):
            raise ValueError(f"Could not find wavs folder at {path_to_wavs_folder}")

    def _filter_metadata_df_on_wav_names(self, wavs_names):
        initial_row_count = len(self.metadata_df)
        self.metadata_df = self.metadata_df[
            self.metadata_df["file_name"].isin(wavs_names)
        ]
        removed_rows = initial_row_count - len(self.metadata_df)
        if removed_rows:
            logging.info(
                f"Removed {removed_rows} rows from metadata_df because the corresponding WAV files were not found."
            )

    # List of functions which builds the pipeline / recipe for the data
    def set_window_size(self, window_size_in_seconds: int = 1):
        """
        Set the window size for the data
        """
        assert type(window_size_in_seconds) == int, "Window size must be an integer"
        self.pipeline.window_size = window_size_in_seconds

    def set_label_to_class_map(self, label_to_class_map: Dict = None):
        """
        Set the mapping from label to class for the data
        """
        assert (
            type(label_to_class_map) == dict
        ), "Label to class map must be a dictionary"
        self.pipeline.label_to_class_map = label_to_class_map

    def set_augmentations(self, augmentations: List = None):
        """
        Set the augmentation steps for the data
        """
        assert type(augmentations) == list, "Augmentations must be a list"
        assert all(
            augmentation in Augmenter.augment_options for augmentation in augmentations
        ), f"Some augmentations not in possible augmentations {Augmenter.augment_options}"

        self.pipeline.augmentations = augmentations

    def set_audio_format(self, audio_format: str = "stft"):
        """
        Set the audio format for the data
        """
        assert str(audio_format), "Audio format must be a string"
        assert (
            audio_format in AudioFormatter.audio_format_options
        ), "Audio format not supported"
        self.pipeline.audio_format = audio_format

    def set_sample_rate(self, sample_rate: int = 44100):
        """
        Set the sample rate for the data
        """
        assert type(sample_rate) == int, "Sample rate must be an integer"
        self.pipeline.sample_rate = sample_rate

    def set_split_configuration(
        self,
        train_percent: int = 70,
        test_percent: int = 15,
        validation_percent: int = 15,
    ):
        """
        Set the split for the data
        """
        if train_percent + test_percent + validation_percent != 100:
            raise ValueError("Split percentages must add up to 100")
        self.pipeline.split = {
            "train": train_percent,
            "test": test_percent,
            "validation": validation_percent,
        }

    def set_file_type(self, file_type: str = "tfrecord"):
        """
        Set the file type for the data
        """
        assert type(file_type) == str, "File type must be a string"
        assert file_type in FileTyper.file_type_options, "File type not supported"
        self.pipeline.file_type = file_type

    def set_limit(self, limit: int = None):  # TODO: Not finished
        """
        Set the limit the number of files for each split
        """
        assert type(limit) == int, "Limit must be an integer"
        if not isinstance(limit, int):
            raise ValueError(f"Limit must be an integer")
        self.pipeline.limit = limit

    # List of functions to describe or perform the pipeline / recipe for the data
    def describe_it(self):
        """
        Describe the data / pipeline / recipe for the data
        """
        self.pipeline.describe(self.metadata_df)

    def make_it(self, clean: bool = False):
        """
        Run the pipeline / recipe for the data
        """
        assert type(clean) == bool, "Clean must be a boolean"
        self.pipeline.make(self.metadata_df, clean=clean)

    def load_it(self, label_encoding="integer"):
        """
        Load the data.
        args:
            label_encoding: str
                The label encoding format. Options: 'integer', 'one_hot', 'binary'
        """
        assert label_encoding in [
            "integer",
            "one_hot",
            "binary",
        ], "Label format not supported"
        return self.dataloader.load(label_encoding)
