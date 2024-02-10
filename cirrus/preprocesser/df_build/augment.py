import pandas as pd
from typing import List


def augment(df: pd.DataFrame, augmentations: List):
    df["augmentation"] = None
    # Filter out test rows, as we don't augment them
    non_test_df = df[df["split"] != "test"]

    # Create a DataFrame for each augmentation
    augmented_dfs = [non_test_df.copy() for _ in augmentations]

    # Assign the augmentation labels
    for aug_df, augmentation in zip(augmented_dfs, augmentations):
        aug_df["augmentation"] = augmentation

    # Concatenate all augmented DataFrames along with the original non-test DataFrame
    result_df = pd.concat(augmented_dfs + [df], ignore_index=True)

    return result_df
