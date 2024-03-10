import pandas as pd
import os
import subprocess
import numpy as np
import librosa
from tqdm import tqdm
from ..audio_formatter.audio_formatter import AudioFormatter
from ..augmenter.augmenter import Augmenter
from ..file_typer.file_typer import FileTyper

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def pre_preprocess(df: pd.DataFrame, data_output_path: str, clean: bool):
    logging.info("Doing pre-preprocessing..")
    # Check if data_output_path exists (Here the wavs are going to be saved)

    if clean and os.path.exists(data_output_path):
        logging.info("Cleaning data_output_path")
        subprocess.call(["rm", "-rf", data_output_path])

    logging.info("Checking if data_output_path exists. If not, creating it")
    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)

    # print("Getting files in data_output_path")
    # data_output_path_contents = os.listdir(data_output_path)

    wavs_to_pipeline_df = df  # THis is a temporary solution
    # print("Stripping the file extension")
    # # Strip the file extension
    # data_output_path_contents_hash = [
    #     file.split(".")[0] for file in data_output_path_contents
    # ]

    # print("Removing all files in df which are already in data_output_path")
    # # Remove all files in df which are already in data_output_path
    # len_before = len(df)
    # wavs_to_pipeline_df = df[~df["hash"].isin(data_output_path_contents_hash)]
    # len_after = len(wavs_to_pipeline_df)
    # if len_before != len_after:
    #     logging.info(
    #         "Skipping %s files which are already pipelined and saved in the data_output_path",
    #         len_before - len_after,
    #     )
    return wavs_to_pipeline_df


def get_wav_chunk(
    wav: np.ndarray, start: int, end: int, sample_rate: int, wav_length: int
):
    assert end <= wav_length, "Trying to create window which exceeds the wav's lenght"
    wav_chunk = wav[int(start * sample_rate) : int(end * sample_rate)]
    return wav_chunk


def preprocess(df: pd.DataFrame, input_path: str, output_path: str):
    logging.info("Doing preprocessing..")
    df = df.sort_values("file_name")
    df = df.reset_index(drop=True)

    augmenter = Augmenter()
    audio_formatter = AudioFormatter()
    file_typer = FileTyper()
    wav_currently_read = None
    length_of_current_wav = None
    wav = None
    sample_rate = None
    shape_validation = None
    for _, row in tqdm(
        df.iterrows(), total=df.shape[0], ncols=100, desc="Making dataset"
    ):
        # Check if the file is already in the output path. If so, continue
        if os.path.exists(
            os.path.join(output_path, row["hash"] + "." + row["file_type"])
        ):
            continue

        # Read new wav if necessary
        if wav_currently_read != row["file_name"]:
            wav_currently_read = row["file_name"]
            wav, sample_rate = librosa.load(
                os.path.join(input_path, "wavs", wav_currently_read), sr=44100
            )
            length_of_current_wav = len(wav)

        # Make a chunk of the wav
        wav_chunk = get_wav_chunk(
            wav,
            row["label_relative_start_sec"],
            row["label_relative_end_sec"],
            sample_rate,
            length_of_current_wav,
        )

        if row.get("augmentation") in augmenter.augment_options:
            wav_chunk = augmenter.augment_file(
                wav_chunk, sample_rate, row.get("augmentation")
            )

        wav_chunk = audio_formatter.audio_format_file(
            wav_chunk, sample_rate, row["audio_format"]
        )

        # Validating that all outputted files have the same shape
        if shape_validation is None:
            shape_validation = wav_chunk.shape
        assert (
            shape_validation == wav_chunk.shape
        ), "All outputted files must have the same shape"

        file_typer.save_file(
            wav_chunk, output_path, row["hash"], row["file_type"], row["audio_format"]
        )


def post_preprocess(df: pd.DataFrame, data_info_output_path: str):
    logging.info("Doing post preprocessing..")
    if not os.path.exists(data_info_output_path):
        os.makedirs(data_info_output_path)
    df = df[["hash", "class", "split"]]
    df.to_csv(os.path.join(data_info_output_path, "dataset.csv"), index=False)


def make(df: pd.DataFrame, data_input_path: str, data_output_path: str, clean=False):
    logging.info("Making..")
    assert len(df) > 0, "Dataframe is empty"
    assert "file_name" in df.columns, "Dataframe does not contain 'file_name' column"

    wavs_to_pipeline_df = pre_preprocess(df, data_output_path, clean)
    preprocess(wavs_to_pipeline_df, data_input_path, data_output_path)
    post_preprocess(df, "cache")
