import pandas as pd
import json


def save_preprocessing_config(
    df: pd.DataFrame,
    window_size: int,
    label_map: dict,
    augmentations: list,
    audio_format: str,
    sample_rate: int,
    split: dict,
    file_type: str,
    limit: int,
    remove_labels: list,
    overlap_threshold: float,
):
    """Save the preprocessing configuration to a json file."""

    # Create a dict with the split information (length of each split)
    split_info_dict = {}
    for split in df["split"].unique():
        split_df = df[df["split"] == split]
        split_info_dict[split] = len(split_df)

    preprocessing_config = {
        "window_size": window_size,
        "label_map": label_map,
        "augmentations": augmentations,
        "audio_format": audio_format,
        "sample_rate": sample_rate,
        "split": split,
        "file_type": file_type,
        "limit": limit,
        "remove_labels": remove_labels,
        "overlap_threshold": overlap_threshold,
        "split_info": split_info_dict,
    }
    path_to_file = "cache/preprocessing_config.json"
    with open(path_to_file, "w") as file:
        json.dump(preprocessing_config, file, indent=4)
