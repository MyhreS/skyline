import os

import imageio
import numpy as np


def get_path_to_image(image_dir_path: str, split: str, class_: str, hash: str):
    return os.path.join(image_dir_path, split, class_, hash + ".png")


def write_as_image(
    wav_chunk: np.ndarray,
    output_path: str,
    split: str,
    class_: str,
    hash: str,
    shape_validation,
):
    image_output_path = get_path_to_image(output_path, split, class_, hash)
    # S_scaled = np.clip(
    #     (wav_chunk - wav_chunk.min()) / (wav_chunk.max() - wav_chunk.min()) * 255,
    #     0,
    #     255,
    # ).astype(np.uint8)

    S_scaled = np.clip(
        (wav_chunk - wav_chunk.min()) / (wav_chunk.max() - wav_chunk.min()) * 65535,
        0,
        65535,
    ).astype(
        np.uint16
    )  # Use uint16 for 16-bit depth

    if not os.path.exists(os.path.dirname(image_output_path)):
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)

    if S_scaled.size == 0:
        print(
            f"Scaled array (S_scaled) is empty after processing for: {image_output_path}"
        )
        return

    if np.isnan(S_scaled).any() or np.isinf(S_scaled).any():
        print(f"Scaled array (S_scaled) contains NaNs or Infs for: {image_output_path}")
        return

    if S_scaled.shape != shape_validation:
        print(
            f"Scaled array (S_scaled) has incorrect shape for: {image_output_path} which is {S_scaled.shape} and should be {shape_validation}"
        )
        return

    imageio.imwrite(image_output_path, S_scaled)
