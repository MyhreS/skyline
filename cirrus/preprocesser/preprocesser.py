from .df_build.window import window
from .df_build.train_val_test_split import train_val_test_split
from .df_build.map_label_to_class import map_label_to_class
from .df_build.hash import hash
from .df_build.limit import limit
from .make.make import make

from ..utils.augmenter.augmenter import Augmenter
from ..utils.audio_formatter.audio_formatter import AudioFormatter
from ..utils.file_typer.file_typer import FileTyper

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


class Preprocesser:
    def __init__(self, data_input_path: str, data_output_path: str):
        self.data_input_path = data_input_path
        self.data_output_path = data_output_path

        self.window_size = None
        self.label_to_class_map = None
        self.augmentations = None
        self.audio_format = None
        self.sample_rate = None
        self.split = None
        self.file_type = None
        self.limit = None
        self.overlap_threshold = 1.0

    def _build(self, df: pd.DataFrame):
        logging.info("Building dataframe representation of the data.")
        df = window(df, self.window_size, self.overlap_threshold)
        # Check that self.split['train'] exists:
        df = train_val_test_split(
            df, self.split["train"], self.split["test"], self.split["validation"]
        )
        if self.label_to_class_map is not None:
            df = map_label_to_class(df, self.label_to_class_map)
        else:
            df["class"] = df["label"]
        df["sample_rate"] = self.sample_rate
        df = Augmenter().augment_df_files(df, self.augmentations)
        if self.limit is not None:
            df = limit(df, self.limit)

        df = AudioFormatter().audio_format_df_files(df, self.audio_format)
        df = FileTyper().file_type_df_files(df, self.file_type)
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
        manipulated_length = len(df)
        manipulated_duration = df["label_duration_sec"].sum()
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
            print(f"|| Split {split} of length {len(split_df)}")
            for class_ in split_df["class"].unique():
                class_df = split_df[split_df["class"] == class_]
                print(f"|||| Class {class_} of length {len(class_df)}")
                for label in class_df["label"].unique():
                    label_df = class_df[class_df["label"] == label]
                    print(f"|||||| Label {label} of length {len(label_df)}")

    def _df_validation(self, df: pd.DataFrame):
        assert len(df) > 0, "Dataframe is empty"
        # assert len(df["class"].unique()) > 1, "Dataframe contains only one class"

    def make(self, df: pd.DataFrame, clean=False):
        logging.info("Running preprocesser to create the dataset.")
        build_df = self._build(df)
        self._df_validation(build_df)
        make(build_df, self.data_input_path, self.data_output_path, clean=clean)
