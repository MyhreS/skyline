import pandas as pd
import os
import subprocess
import numpy as np
import librosa
from tqdm import tqdm
from ..audio_formatter.audio_formatter import AudioFormatter
from ..augmenter.augmenter import Augmenter
from .write_as_tfrecord import write_as_tfrecord

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def pre_preprocess(df: pd.DataFrame, data_output_path: str, clean: bool):
    logging.info("Doing pre-datamaking..")
    # Check if data_output_path exists (Here the wavs are going to be saved)

    if clean and os.path.exists(data_output_path):
        logging.info("Cleaning data_output_path")
        subprocess.call(["rm", "-rf", data_output_path])

    logging.info("Checking if data_output_path exists. If not, creating it")
    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)
    wavs_to_pipeline_df = df
    return wavs_to_pipeline_df


def get_wav_chunk(
    wav: np.ndarray, start: int, end: int, sample_rate: int, wav_length: int
):
    assert end <= wav_length, "Trying to create window which exceeds the wav's lenght"
    wav_chunk = wav[int(start * sample_rate) : int(end * sample_rate)]
    return wav_chunk


def preprocess(df: pd.DataFrame, input_path: str, output_path: str):
    logging.info("Doing datamaking..")
    df = df.sort_values("file_name")
    df = df.reset_index(drop=True)

    augmenter = Augmenter()
    audio_formatter = AudioFormatter()
    wav_currently_read = None
    length_of_current_wav = None
    wav = None
    shape_validation = None
    for index, row in tqdm(
        df.iterrows(), total=df.shape[0], ncols=100, desc="Making dataset"
    ):
        # Check if the file is already in the output path. If so, continue
        if os.path.exists(os.path.join(output_path, row["hash"] + ".tfrecord")):
            continue

        # Read new wav if necessary
        if wav_currently_read != row["file_name"]:
            wav_currently_read = row["file_name"]
            try:
                wav, sr = librosa.load(
                    os.path.join(input_path, "wavs", wav_currently_read), sr=16000
                )
            except Exception as e:
                logging.error(f"Error loading {wav_currently_read}: {e}")
                # Delete the problematic file
                os.remove(os.path.join(input_path, "wavs", wav_currently_read))
                # Remove the row from the dataframe
                df = df.drop(index)
                continue  # Skip to the next iteration

            length_of_current_wav = len(wav)

        # Make a chunk of the wav
        wav_chunk = get_wav_chunk(
            wav,
            row["window_start_in_sec"],
            row["window_end_in_sec"],
            sr,
            length_of_current_wav,
        )

        if row.get("augmentation") in augmenter.augment_options:
            wav_chunk = augmenter.augment_file(wav_chunk, sr, row.get("augmentation"))

        wav_chunk = audio_formatter.audio_format_file(
            wav_chunk, sr, row["audio_format"]
        )

        # Validating that all outputted files have the same shape
        if shape_validation is None:
            shape_validation = wav_chunk.shape
        assert (
            shape_validation == wav_chunk.shape
        ), "All outputted files must have the same shape"

        write_as_tfrecord(wav_chunk, output_path, row["hash"], row["audio_format"])


def post_preprocess(df: pd.DataFrame, data_info_output_path: str):
    logging.info("Doing post datamaking..")
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
