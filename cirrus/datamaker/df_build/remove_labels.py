import pandas as pd
from typing import List

"""
Function for removing labels from a dataframe
"""


def remove_labels(df: pd.DataFrame, remove_labels: List) -> pd.DataFrame:
    """Remove a label from the data (however not from test)"""
    assert len(df) > 0, "Dataframe is empty"
    assert "label" in df.columns, "Dataframe does not contain 'label' column"
    assert "split" in df.columns, "Dataframe does not contain 'split' column"

    test_split_df = df[df["split"].str.contains("test")]
    df = df[~df["split"].str.contains("test")]

    for label in remove_labels:
        df = df[df["label"] != label]

    return pd.concat([df, test_split_df])
