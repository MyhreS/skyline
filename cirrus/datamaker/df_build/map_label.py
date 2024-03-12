import pandas as pd
from typing import Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def _map_label_to_class(row: pd.Series, label_to_class_map: Dict):
    for class_name, labels in label_to_class_map.items():
        if row["label"] in labels:
            row["class"] = class_name
            return row
    return row


def map_label(df: pd.DataFrame, label_to_class_map: Dict):
    assert "label" in df.columns, "df must contain column 'label'"
    if label_to_class_map is None:
        df["class"] = df["label"]
        return df

    df["class"] = None
    mapped_df = df.apply(
        lambda row: _map_label_to_class(row, label_to_class_map), axis=1
    )

    # Remove the rows where class is None
    unmapped_df = mapped_df[mapped_df["class"].isna()]
    if len(unmapped_df) > 0:
        unmapped_labels = unmapped_df["label"].unique()
        logging.warning(
            f"The following labels could not be mapped to a class and where therefore removed: {unmapped_labels}"
        )
    mapped_df = mapped_df[mapped_df["class"].notna()]
    if len(unmapped_df) > 0:
        remaining_labels = mapped_df["label"].unique()
        logging.info(f"The following labels were mapped to a class: {remaining_labels}")
    assert len(mapped_df) > 0, "Mapped dataframe is empty"
    return mapped_df
