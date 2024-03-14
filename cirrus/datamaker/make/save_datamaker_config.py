import pandas as pd
import json
import os


def save_datamaker_config(
    df: pd.DataFrame,
    window_size: int,
    label_map: dict,
    augmentations: list,
    audio_format: str,
    original_sample_rate: int,
    val_split: float,
    limit: int,
    remove_labels: list,
    overlap_threshold: float,
):
    """Save the datamaker configuration to a json file."""

    # Create a dict with the split information (length of each split)
    split_info_dict = {}
    for split in df["split"].unique():
        split_df = df[df["split"] == split]
        split_info_dict[split] = len(split_df)

    datamaker_config = {
        "window_size": window_size,
        "label_map": label_map,
        "augmentations": augmentations,
        "audio_format": audio_format,
        "original_sample_rate": original_sample_rate,
        "new_sample_rate": 16000,
        "val_split": val_split,
        "limit": limit,
        "remove_labels": remove_labels,
        "overlap_threshold": overlap_threshold,
        "split_info": split_info_dict,
    }
    if not os.path.exists("cache"):
        os.makedirs("cache")
    path_to_file = "cache/datamaker_config.json"
    with open(path_to_file, "w") as file:
        json.dump(datamaker_config, file, indent=4)
