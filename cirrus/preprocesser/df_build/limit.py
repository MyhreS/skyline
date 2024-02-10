import pandas as pd


def get_class_share(df: pd.DataFrame, limit: int) -> dict:
    # Count the occurrences of each label
    label_counts = df["class"].value_counts()

    # Calculate total rows to select, ensuring it does not exceed the limit
    total_rows = min(limit, len(df))

    # Calculate initial share for each label based on the limit
    label_share = {
        label: min(count, round(total_rows / len(label_counts)))
        for label, count in label_counts.items()
    }

    # Adjust the share based on the limit and the initial calculation
    while sum(label_share.values()) != total_rows:
        # Calculate the difference between desired total rows and current sum of label shares
        diff = total_rows - sum(label_share.values())

        if diff > 0:
            # If there's room to add more rows, distribute them to the labels with space left
            for label, count in label_counts.items():
                if label_share[label] < count and diff > 0:
                    label_share[label] += 1
                    diff -= 1
                    if diff <= 0:
                        break
        else:
            # If there are too many rows, remove from the largest share(s) first
            largest_labels = sorted(label_share, key=label_share.get, reverse=True)
            for label in largest_labels:
                if label_share[label] > 0:
                    label_share[label] -= 1
                    diff += 1
                    if diff >= 0:
                        break
    return label_share


def get_label_share(df: pd.DataFrame, limit: int) -> dict:
    # Count the occurrences of each label
    label_counts = df["label"].value_counts()

    # Calculate total rows to select, ensuring it does not exceed the limit
    total_rows = min(limit, len(df))

    # Calculate initial share for each label based on the limit
    label_share = {
        label: min(count, round(total_rows / len(label_counts)))
        for label, count in label_counts.items()
    }

    # Adjust the share based on the limit and the initial calculation
    while sum(label_share.values()) != total_rows:
        # Calculate the difference between desired total rows and current sum of label shares
        diff = total_rows - sum(label_share.values())

        if diff > 0:
            # If there's room to add more rows, distribute them to the labels with space left
            for label, count in label_counts.items():
                if label_share[label] < count and diff > 0:
                    label_share[label] += 1
                    diff -= 1
                    if diff <= 0:
                        break
        else:
            # If there are too many rows, remove from the largest share(s) first
            largest_labels = sorted(label_share, key=label_share.get, reverse=True)
            for label in largest_labels:
                if label_share[label] > 0:
                    label_share[label] -= 1
                    diff += 1
                    if diff >= 0:
                        break
    return label_share


def limit_labeles_of_class(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    assert len(df) > 0, "Dataframe is empty"
    assert "label" in df.columns, "Dataframe does not contain 'class' column"

    label_share = get_label_share(df, limit)
    # Sample rows based on the calculated share for each label
    sampled_dfs = [
        df[df["label"] == label].sample(n=label_share[label], random_state=1)
        for label in label_share
    ]

    # Concatenate the sampled dataframes back into one
    result_df = pd.concat(sampled_dfs).reset_index(drop=True)

    return result_df


def limit_classes_of_split(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    assert len(df) > 0, "Dataframe is empty"
    assert "class" in df.columns, "Dataframe does not contain 'class' column"

    class_shares = get_class_share(df, limit)

    list_of_dfs = []
    for class_share in class_shares:
        df_class = df[df["class"] == class_share]
        labels_limited_of_class = limit_labeles_of_class(
            df_class, class_shares[class_share]
        )
        list_of_dfs.append(labels_limited_of_class)

    result_df = pd.concat(list_of_dfs).reset_index(drop=True)
    return result_df


def limit(
    df: pd.DataFrame, limit: int
):  # TODO: Sampe on duration. And sample on duration sec
    assert len(df) > 0, "Dataframe is empty"
    assert "split" in df.columns, "Dataframe does not contain 'split' column"

    if limit > len(df):
        return df

    test_split_percentage = len(df[df["split"] == "test"]) / len(df)
    train_split_percentage = len(df[df["split"] == "train"]) / len(df)
    val_split_percentage = len(df[df["split"] == "validation"]) / len(df)

    test_limit = int(limit * test_split_percentage)
    train_limit = int(limit * train_split_percentage) + int(test_limit / 2)
    val_limit = int(limit * val_split_percentage) + int(test_limit / 2)
    train_df = limit_classes_of_split(df[df["split"] == "train"], train_limit)
    val_df = limit_classes_of_split(df[df["split"] == "validation"], val_limit)

    test_df = df[df["split"] == "test"]

    result_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    return result_df
