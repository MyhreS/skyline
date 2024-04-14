import os
import pandas as pd
from tqdm import tqdm


def create_row(
    sqbundle_id,
    file_name,
    file_duration,
    label,
    window_start,
    window_end,
    split,
):
    return {
        "sqbundle_id": sqbundle_id,
        "file_name": file_name,
        "file_duration_in_sec": file_duration,
        "window_start_in_sec": window_start,
        "window_end_in_sec": window_end,
        "window_duration_in_sec": window_end - window_start,
        "label": label,
        "split": split,
    }


def calculate_overlap(window_start, window_end, label_start, label_end):
    start = max(label_start, window_start)
    end = min(label_end, window_end)
    overlap_duration = max(0, end - start)
    return overlap_duration


def get_label_in_window(file_df, window_start, window_end, window_overlap_threshold):
    for _, row in file_df.iterrows():

        label_duration = row["label_relative_end_sec"] - row["label_relative_start_sec"]
        file_duration = row["wav_duration_sec"]

        # Check if the entire file is a single label
        if label_duration == file_duration:
            return row["label"]

        # Calculate overlap for non-full-length labels
        overlap_duration = calculate_overlap(
            window_start,
            window_end,
            row["label_relative_start_sec"],
            row["label_relative_end_sec"],
        )
        if overlap_duration > window_overlap_threshold:
            return row["label"]
    return None


def window_file(file_df, window_size, window_overlap_threshold, window_overlap_step):
    sqbundle_id = file_df["sqbundle_id"].iloc[0]
    file_name = file_df["file_name"].iloc[0]
    file_duration = file_df["wav_duration_sec"].iloc[0]
    split = file_df["split"].iloc[0]

    windowed_rows = []
    window_start = 0
    window_end = window_size
    while window_end <= file_duration:
        label_in_window = get_label_in_window(
            file_df, window_start, window_end, window_overlap_threshold
        )
        if label_in_window:
            windowed_rows.append(
                create_row(
                    sqbundle_id,
                    file_name,
                    file_duration,
                    label_in_window,
                    window_start,
                    window_end,
                    split,
                )
            )
        window_start += window_overlap_step
        window_end += window_overlap_step
    return pd.DataFrame(windowed_rows)


def window_files(df, window_size, window_overlap_threshold, window_overlap_step):
    windowed_files = []

    tut_dcase = df[df["label"] == "TUT_dcase"]
    non_tut_dcase = df[df["label"] != "TUT_dcase"]

    non_tut_dcase = non_tut_dcase[
        ~non_tut_dcase["file_name"].isin(tut_dcase["file_name"].unique())
    ]

    tut_dcase.reset_index(drop=True, inplace=True)
    non_tut_dcase.reset_index(drop=True, inplace=True)

    df = pd.concat([tut_dcase, non_tut_dcase])

    for file_name in tqdm(df["file_name"].unique(), desc="Windowing status", ncols=100):
        file_df = df[df["file_name"] == file_name]

        windowed_files.append(
            window_file(
                file_df, window_size, window_overlap_threshold, window_overlap_step
            )
        )
    return pd.concat(windowed_files)


def window(
    df,
    window_size,
    window_overlap_threshold,
    window_overlap_step,
    load_cached_windowing=False,
):
    """
    Returns a dataframe with columns: sqbundle_id, file_name, wav_duration_sec, window_start, window_end, label
    """
    assert len(df) > 0, "df must be non-empty"
    print("Unique labels: ", df["label"].unique())
    for _, row in df.iterrows():
        if row["label_relative_end_sec"] > row["wav_duration_sec"]:
            raise ValueError(
                f"Label end time is later than wav duration, label end: {row['label_relative_end_sec']}, wav duration: {row['wav_duration_sec']}"
            )

    if load_cached_windowing and os.path.exists("cache/windowed_files.csv"):
        windowed_files = pd.read_csv("cache/windowed_files.csv")
    else:
        windowed_files: pd.DataFrame = window_files(
            df, window_size, window_overlap_threshold, window_overlap_step
        )
        windowed_files.reset_index(drop=True, inplace=True)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        windowed_files.to_csv("cache/windowed_files.csv", index=False)

    assert len(windowed_files) > 0, "Windowed dataframe is empty after windowing"
    return windowed_files
