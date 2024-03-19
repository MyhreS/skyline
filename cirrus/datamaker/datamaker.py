import json
import os
from .df_build.window import window
from .df_build.split import split
from .df_build.map_label import map_label
from .df_build.sample_rate import sample_rate
from .df_build.hash import hash
from .df_build.limit import limit
from .df_build.remove_labels import remove_labels
from .make.make import make

from .augmenter.augmenter import Augmenter
from .audio_formatter.audio_formatter import AudioFormatter

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def _create_summary_dict(df: pd.DataFrame, run_id: str):
    summary_dict = {}

    for split in df["split"].unique():
        split_dict = {}
        split_df = df[df["split"] == split]
        for class_ in split_df["class"].unique():
            class_dict = {}
            class_df = split_df[split_df["class"] == class_]
            for label in class_df["label"].unique():
                label_df = class_df[class_df["label"] == label]
                class_dict[label] = len(label_df)
            split_dict[class_] = class_dict
        summary_dict[split] = split_dict

    path_to_output_data_summary = os.path.join("cache", run_id, "data_summary.json")
    if not os.path.exists(os.path.dirname(path_to_output_data_summary)):
        os.makedirs(os.path.dirname(path_to_output_data_summary))
    with open(path_to_output_data_summary, "w") as file:
        json.dump(summary_dict, file, indent=5)
    return summary_dict


class Datamaker:
    def __init__(self, data_input_path: str, data_output_path: str, run_id: str):
        self.data_input_path = data_input_path
        self.data_output_path = data_output_path

        self.window_size = 1
        self.overlap_threshold = 1
        self.overlap_steps = 0.5
        self.load_cached_windowing = False
        self.label_map = None
        self.augmentations = None
        self.only_augment_drone = False
        self.audio_format = None
        self.original_sample_rate = None
        self.limit = None
        self.save_format = None

        self.val_percent = 0.2
        self.remove_labels = []

        self.build_df = []

        self.run_id = run_id

    def _build(self, df: pd.DataFrame):
        logging.info("Building dataframe representation of the data.")
        logging.info("Windowing..")
        df = window(
            df,
            self.window_size,
            self.overlap_threshold,
            self.overlap_steps,
            load_cached_windowing=self.load_cached_windowing,
        )
        logging.info("Splitting..")
        df = split(df, self.val_percent)
        logging.info("Mapping labels..")
        df = map_label(df, self.label_map)
        logging.info("Removing labels..")
        df = remove_labels(df, remove_labels=self.remove_labels)
        logging.info("Setting sample rate..")
        df = sample_rate(df, self.original_sample_rate)
        logging.info("Augmenting..")
        df = Augmenter(path_to_input_data=self.data_input_path).augment_df_files(
            df, self.augmentations, self.only_augment_drone
        )
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
        summary_dict = _create_summary_dict(df, self.run_id)
        print(json.dumps(summary_dict, indent=4))

    def _df_validation(self, df: pd.DataFrame):
        assert len(df) > 0, "Dataframe is empty"
        # assert len(df["class"].unique()) > 1, "Dataframe contains only one class"

    def make(self, clean=False):
        logging.info("Running preprocesser to create the dataset.")
        assert (
            len(self.build_df) > 0
        ), "Build dataframe datarepresentation is empty. Run describe() first."
        self._df_validation(self.build_df)
        self._save_config()
        make(
            self.build_df,
            self.data_input_path,
            self.data_output_path,
            self.label_map,
            self.save_format,
            clean=clean,
        )

    def _save_config(self):
        config = (
            {
                "data_input_path": self.data_input_path,
                "data_output_path": self.data_output_path,
                "window_size": self.window_size,
                "overlap_threshold": self.overlap_threshold,
                "overlap_steps": self.overlap_steps,
                "load_cached_windowing": self.load_cached_windowing,
                "label_map": self.label_map,
                "augmentations": self.augmentations,
                "only_augment_drone": self.only_augment_drone,
                "audio_format": self.audio_format,
                "original_sample_rate": self.original_sample_rate,
                "limit": self.limit,
                "save_format": self.save_format,
                "val_percent": self.val_percent,
                "remove_labels": self.remove_labels,
                "length_of_build_df": len(self.build_df),
                "unique_splits": self.build_df["split"].unique().tolist(),
                "unique_classes": self.build_df["class"].unique().tolist(),
                "unique_labels": self.build_df["label"].unique().tolist(),
                "window_duration_in_sec": self.build_df["window_duration_in_sec"].sum(),
            },
        )

        path_to_output_config = os.path.join(
            "cache", self.run_id, "datamaker_config.json"
        )
        if not os.path.exists(os.path.dirname(path_to_output_config)):
            os.makedirs(os.path.dirname(path_to_output_config))
        with open(path_to_output_config, "w") as file:
            json.dump(config, file, indent=4)
