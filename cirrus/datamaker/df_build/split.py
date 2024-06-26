import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)

"""
Functions for splitting the data into train, val and test.
"""


def split(df, val_percent):
    assert "label" in df.columns, "df must contain column 'label'"
    assert "sqbundle_id" in df.columns, "df must contain column 'sqbundle_id'"
    assert val_percent < 0.99, "val_percent must be less than 99"
    assert "split" in df.columns, "df must contain column 'split'"

    return set_val(df, 0.2)


def set_val(df, val_percent):
    df = df.copy()
    indices_to_update = []
    for label in df["label"].unique():
        non_test = df[(~df["split"].str.contains("test")) & (df["label"] == label)]
        num_rows = int(len(non_test) * val_percent)
        selected_indices = non_test.iloc[:num_rows].index
        indices_to_update.extend(selected_indices)
    df.loc[indices_to_update, "split"] = "val"
    return df
