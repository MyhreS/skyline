from .df_build.window import window
from .df_build.split import split
from .df_build.map_label import map_label
from .df_build.sample_rate import sample_rate
from .df_build.hash import hash
from .df_build.limit import limit
from .df_build.remove_labels import remove_labels
from .make.make import make
from .make.save_datamaker_config import save_datamaker_config

from .augmenter.augmenter import Augmenter
from .audio_formatter.audio_formatter import AudioFormatter

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


class Datamaker:
    def __init__(self, data_input_path: str, data_output_path: str):
        self.data_input_path = data_input_path
        self.data_output_path = data_output_path

        self.window_size = None
        self.label_map = None
        self.augmentations = None
        self.audio_format = None
        self.original_sample_rate = None
        self.limit = None
        self.overlap_threshold = 1.0
        self.val_percent = 0.2
        self.remove_labels = []

        self.build_df = []

    def _build(self, df: pd.DataFrame):
        logging.info("Building dataframe representation of the data.")
        logging.info("Windowing..")
        df = window(df, self.window_size, self.overlap_threshold)
        logging.info("Splitting..")
        df = split(df, self.val_percent)
        logging.info("Mapping labels..")
        df = map_label(df, self.label_map)
        logging.info("Removing labels..")
        df = remove_labels(df, remove_labels=self.remove_labels)
        logging.info("Setting sample rate..")
        df = sample_rate(df, self.original_sample_rate)
        logging.info("Augmenting..")
        df = Augmenter().augment_df_files(df, self.augmentations)
        logging.info("Limiting..")
        df = limit(df, self.limit)
        logging.info("Setting audio format..")
        df = AudioFormatter().audio_format_df_files(df, self.audio_format)
        logging.info("Hashing..")
        df = hash(df)
        return df

    def describe(self, df: pd.DataFrame):
        logging.info("Describing data:")
        original_length = len(df)
        original_duration = df["label_duration_sec"].sum()
        original_column_names = df.columns
        logging.info(
            "Found original data of length %s, duration %s and columns %s",
            original_length,
            original_duration,
            list(original_column_names),
        )

        df = self._build(df)
        assert len(df) > 0, "Dataframe is empty"
        assert "split" in df.columns, "Dataframe does not contain 'split' column"
        assert "class" in df.columns, "Dataframe does not contain 'class' column"
        assert "label" in df.columns, "Dataframe does not contain 'label' column"
        self.build_df = df
        manipulated_length = len(df)
        manipulated_duration = df["window_duration_in_sec"].sum()
        manipulated_column_names = df.columns
        logging.info(
            "Found pipelined data of length %s, duration %s and columns %s:",
            manipulated_length,
            manipulated_duration,
            list(manipulated_column_names),
        )
        logging.info("Pipelined data is split into:")
        for split in df["split"].unique():
            split_df = df[df["split"] == split]
            print(f"Split {split} of length {len(split_df)}")
            for class_ in split_df["class"].unique():
                class_df = split_df[split_df["class"] == class_]
                print(f"   Class {class_} of length {len(class_df)}")
                for label in class_df["label"].unique():
                    label_df = class_df[class_df["label"] == label]
                    print(f"      Label {label} of length {len(label_df)}")

    def _df_validation(self, df: pd.DataFrame):
        assert len(df) > 0, "Dataframe is empty"
        # assert len(df["class"].unique()) > 1, "Dataframe contains only one class"

    def make(self, df: pd.DataFrame, clean=False):
        logging.info("Running preprocesser to create the dataset.")
        assert (
            len(self.build_df) > 0
        ), "Build dataframe datarepresentation is empty. Run describe() first."
        self._df_validation(self.build_df)
        save_datamaker_config(
            self.build_df,
            self.window_size,
            self.label_map,
            self.augmentations,
            self.audio_format,
            self.original_sample_rate,
            self.val_percent,
            self.limit,
            self.remove_labels,
            self.overlap_threshold,
        )
        make(self.build_df, self.data_input_path, self.data_output_path, clean=clean)
