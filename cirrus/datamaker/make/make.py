import json
import pandas as pd
import os
import subprocess
import numpy as np
import librosa
from tqdm import tqdm
from ..audio_formatter.audio_formatter import AudioFormatter
from ..augmenter.augmenter import Augmenter, normalize_audio_energy
from .write_as_tfrecord import write_as_tfrecord
from .write_as_image import write_as_image, get_path_to_image
from matplotlib import pyplot as plt
import shutil
import imageio

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def pre_preprocess(
    df: pd.DataFrame, data_output_path: str, clean: bool, save_format: str
):
    logging.info("Doing pre-datamaking..")
    # Check if data_output_path exists (Here the wavs are going to be saved)
    if save_format != "image":
        if clean and os.path.exists(data_output_path):
            logging.info("Cleaning data_output_path")
            subprocess.call(["rm", "-rf", data_output_path])

        # Check what data is already in the output path
        data_already_in_output = os.listdir(data_output_path)
        data_already_in_output_df = pd.DataFrame(
            data_already_in_output, columns=["hash_with_extension"]
        )
        data_already_in_output_df = data_already_in_output_df.sort_values(
            "hash_with_extension"
        )
        data_already_in_output_df = data_already_in_output_df.reset_index(drop=True)

        df = df.sort_values("hash")
        df = df.reset_index(drop=True)

        # Remove data that is already in the output path
        df = df[
            ~(df["hash"] + ".tfrecord").isin(
                data_already_in_output_df["hash_with_extension"]
            )
        ]

        logging.info("Checking if data_output_path exists. If not, creating it")
        if not os.path.exists(data_output_path):
            os.makedirs(data_output_path)
        wavs_to_pipeline_df = df
        return wavs_to_pipeline_df
    else:
        create_image_dirs(
            data_output_path,
            df["split"].unique(),
            df["class"].unique(),
            clean,
        )
        return df


def create_image_dirs(
    image_dir_path: str, unique_splits: str, unique_classes: str, clean: bool
):
    if clean and os.path.exists(image_dir_path):
        shutil.rmtree(image_dir_path)
    for unique_split in unique_splits:
        for unique_class in unique_classes:
            path_to_class = os.path.join(
                image_dir_path,
                unique_split,
                unique_class,
            )
            if not os.path.exists(path_to_class):
                os.makedirs(path_to_class)


def get_wav_chunk(
    wav: np.ndarray, start: int, end: int, sample_rate: int, wav_length: int
):
    assert end <= wav_length, "Trying to create window which exceeds the wav's lenght"
    wav_chunk = wav[int(start * sample_rate) : int(end * sample_rate)]
    return wav_chunk


def file_exist(output_path: str, save_format: str, split: str, class_: str, hash: str):
    if save_format != "image":
        if os.path.exists(os.path.join(output_path, hash + ".tfrecord")):
            return True
    else:
        if os.path.exists(
            get_path_to_image(
                output_path,
                split,
                class_,
                hash,
            )
        ):
            return True
    return False


def preprocess(df: pd.DataFrame, input_path: str, output_path: str, save_format: str):
    logging.info("Doing datamaking..")
    df = df.sort_values("file_name")
    df = df.reset_index(drop=True)

    augmenter = Augmenter(path_to_input_data=input_path)
    audio_formatter = AudioFormatter()
    name_of_current_wav = None
    length_of_current_wav = None
    wav = None
    shape_validation = None

    for index, row in tqdm(
        df.iterrows(), total=df.shape[0], ncols=100, desc="Making dataset"
    ):
        if file_exist(
            output_path, save_format, row["split"], row["class"], row["hash"]
        ):
            continue

        # Read new wav if necessary
        if name_of_current_wav != row["file_name"]:
            name_of_current_wav = row["file_name"]
            try:
                wav, sr = librosa.load(
                    os.path.join(input_path, "wavs", name_of_current_wav), sr=16000
                )
            except Exception as e:
                logging.error(f"Error loading {name_of_current_wav}: {e}")
                os.remove(os.path.join(input_path, "wavs", name_of_current_wav))
                df = df.drop(index)
                continue

            length_of_current_wav = len(wav)

        # Make a chunk of the wav
        wav_chunk = normalize_audio_energy(
            get_wav_chunk(
                wav,
                row["window_start_in_sec"],
                row["window_end_in_sec"],
                sr,
                length_of_current_wav,
            )
        )

        if row.get("augmentation") in augmenter.augment_options:
            wav_chunk = augmenter.augment_file(wav_chunk, sr, row.get("augmentation"))

        wav_chunk = audio_formatter.audio_format_file(
            wav_chunk, sr, row["audio_format"]
        )

        if shape_validation is None:
            shape_validation = wav_chunk.shape
            assert (
                shape_validation == wav_chunk.shape
            ), "All outputted files must have the same shape"

        if save_format != "image":
            write_as_tfrecord(wav_chunk, output_path, row["hash"], row["audio_format"])

        else:
            write_as_image(
                wav_chunk,
                output_path,
                row["split"],
                row["class"],
                row["hash"],
                shape_validation,
            )


def post_preprocess(
    df: pd.DataFrame,
    data_info_output_path: str,
    label_class_map: dict,
    save_format: str,
):
    logging.info("Doing post datamaking..")
    if save_format != "image":
        if not os.path.exists(data_info_output_path):
            os.makedirs(data_info_output_path)
        df = df[["hash", "class", "split"]]
        df.to_csv(os.path.join(data_info_output_path, "dataset.csv"), index=False)
        # Save label_class_map
    with open(os.path.join(data_info_output_path, "label_class_map.json"), "w") as f:
        json.dump(label_class_map, f, indent=4)


def make(
    df: pd.DataFrame,
    data_input_path: str,
    data_output_path: str,
    label_map,
    save_format,
    clean=False,
):
    assert len(df) > 0, "Dataframe is empty"
    assert "file_name" in df.columns, "Dataframe does not contain 'file_name' column"

    wavs_to_pipeline_df = pre_preprocess(df, data_output_path, clean, save_format)
    preprocess(wavs_to_pipeline_df, data_input_path, data_output_path, save_format)
    post_preprocess(df, "cache", label_map, save_format)
