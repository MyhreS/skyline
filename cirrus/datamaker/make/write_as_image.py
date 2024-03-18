import os

import imageio
import numpy as np


def get_path_to_image(image_dir_path: str, split: str, class_: str, hash: str):
    return os.path.join(image_dir_path, split, class_, hash + ".png")


def get_path_to_image_dir(data_output_path: str):
    """Gets the path to the image directory, which is on the same level as "data" in the output path"""
    image_dir_path = "/".join(data_output_path.split("/")[:-1])
    return os.path.join(image_dir_path, "image_from_directory")


def write_as_image(
    wav_chunk: np.ndarray, output_path: str, split: str, class_: str, hash: str
):
    image_dir_path = get_path_to_image_dir(output_path)
    image_output_path = get_path_to_image(image_dir_path, split, class_, hash)
    S_scaled = np.clip(
        (wav_chunk - wav_chunk.min()) / (wav_chunk.max() - wav_chunk.min()) * 255,
        0,
        255,
    ).astype(np.uint8)

    imageio.imwrite(image_output_path, S_scaled)
