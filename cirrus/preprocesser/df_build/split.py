import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)


def split(df, train_percent, test_percent, val_percent):
    assert "label" in df.columns, "df must contain column 'label'"
    assert (
        train_percent + test_percent + val_percent == 100
    ), "Split percentages must add up to 100"
    df["split"] = pd.NA
    df = _set_test(df, test_percent)
    df = _set_val(df, val_percent)
    df = _set_train(df)
    return df


def _set_test(df, test_percent):
    assert "split" in df.columns, "df must contain column 'split'"
    smallest_label_size = df.groupby("label").size().min()
    test_size = int(smallest_label_size * (test_percent / 100))
    for label in df["label"].unique():
        label_df = df[df["label"] == label].sort_values(
            by=["wav_duration_sec", "file_name"]
        )
        test_indices = label_df.head(test_size).index
        df.loc[test_indices, "split"] = f"test_{label}"
    file_names_used_for_test = df[df["split"].str.contains("test", na=False)][
        "file_name"
    ].unique()
    length_before_remove = len(df)
    df = df[~((df["file_name"].isin(file_names_used_for_test)) & (df["split"].isna()))]
    length_after_remove = len(df)
    if length_before_remove != length_after_remove:
        logging.warning(
            f"Removed {length_before_remove - length_after_remove} rows from df"
        )
    return df


def _set_val(df, val_percent):
    assert "split" in df.columns, "df must contain column 'split'"
    val_size = int(len(df) * (val_percent / 100))
    not_test_df = df[df["split"].isna()]
    val_indices = not_test_df.sample(n=val_size, random_state=1).index
    df.loc[val_indices, "split"] = "val"
    return df


def _set_train(df):
    assert "split" in df.columns, "df must contain column 'split'"
    df.loc[df["split"].isna(), "split"] = "train"
    return df
