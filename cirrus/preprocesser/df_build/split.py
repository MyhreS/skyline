import pandas as pd


def split(df, train_percent, test_percent, val_percent):
    assert "label" in df.columns, "df must contain column 'label'"
    assert (
        train_percent + test_percent + val_percent == 100
    ), "Split percentages must add up to 100"
    # Create split column and fill with 'train'
    df["split"] = "train"
    smallest_label_size = df.groupby("label").size().min()
    test_amount = int(smallest_label_size * (test_percent / 100))

    for label, group in df.groupby("label"):
        val_amount_label = int(len(group) * (val_percent / 100))
        df.loc[group.index[:test_amount], "split"] = f"test_{label}"
        df.loc[group.index[-val_amount_label:], "split"] = "val"

    return df
