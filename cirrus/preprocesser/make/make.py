import pandas as pd
import os
import shutil
import numpy as np
import librosa
from tqdm import tqdm
from .save_as_npy import save_as_npy
from .save_as_tfrecord import save_as_tfrecord
from ...utils.audio_formatter.audio_formatter import AudioFormatter
from ...utils.augmenter.augmenter import Augmenter

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def pre_preprocess(df: pd.DataFrame, data_output_path: str, clean: bool):
    # Check if data_output_path exists (Here the wavs are going to be saved)
    if clean and os.path.exists(data_output_path):
        shutil.rmtree(data_output_path)

    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)

    data_output_path_contents = os.listdir(data_output_path)
    # Strip the file extension
    data_output_path_contents_hash = [
        file.split(".")[0] for file in data_output_path_contents
    ]

    # Remove all files in df which are already in data_output_path
    len_before = len(df)
    wavs_to_pipeline_df = df[~df["hash"].isin(data_output_path_contents_hash)]
    len_after = len(wavs_to_pipeline_df)
    if len_before != len_after:
        logging.info(
            "Skipping %s files which are already pipelined and saved in the data_output_path",
            len_before - len_after,
        )
    return wavs_to_pipeline_df


def preprocess(df: pd.DataFrame, input_path: str, output_path: str):
    df = df.sort_values("file_name")
    df = df.reset_index(drop=True)

    wav_currently_read = None
    wav = None
    sample_rate = None
    augmenter = Augmenter()
    audio_formatter = AudioFormatter()
    shape_validation = None

    for _, row in tqdm(
        df.iterrows(), total=df.shape[0], ncols=100, desc="Making dataset"
    ):
        # Read new wav if necessary
        if wav_currently_read != row["file_name"]:
            wav_currently_read = row["file_name"]
            wav, sample_rate = librosa.load(
                os.path.join(input_path, "wavs", wav_currently_read), sr=44100
            )
        # Make a chunk of the wav
        wav_chunk = wav[
            int(row["label_relative_start_sec"] * sample_rate) : int(
                row["label_relative_end_sec"] * sample_rate
            )
        ]
        if row.get("augmentation") in augmenter.augment_options:
            wav_chunk_augmented = augmenter.augment_file(
                wav_chunk, sample_rate, row.get("augmentation")
            )
            wav_spectogram = audio_formatter.to_spectrogram(
                wav_chunk_augmented, sample_rate, row["audio_format"]
            )
        else:
            wav_spectogram = audio_formatter.to_spectrogram(
                wav_chunk, sample_rate, row["audio_format"]
            )

        # Validating that all outputted files have the same shape
        if shape_validation is None:
            shape_validation = wav_spectogram.shape
        else:
            assert (
                shape_validation == wav_spectogram.shape
            ), "All outputted files must have the same shape"

        # Save the chunk
        if row["file_type"] == "npy":
            save_as_npy(wav_spectogram, output_path, row["hash"])
        if row["file_type"] == "tfrecord":
            save_as_tfrecord(wav_spectogram, output_path, row["hash"])


def post_preprocess(df: pd.DataFrame, data_info_output_path: str):
    if not os.path.exists(data_info_output_path):
        os.makedirs(data_info_output_path)
    df = df[["hash", "class", "split"]]
    df.to_csv(os.path.join(data_info_output_path, "dataset.csv"), index=False)


def make(df: pd.DataFrame, data_input_path: str, data_output_path: str, clean=False):
    assert len(df) > 0, "Dataframe is empty"
    assert "file_name" in df.columns, "Dataframe does not contain 'file_name' column"

    wavs_to_pipeline_df = pre_preprocess(df, data_output_path, clean)
    preprocess(wavs_to_pipeline_df, data_input_path, data_output_path)
    post_preprocess(df, "cache")
