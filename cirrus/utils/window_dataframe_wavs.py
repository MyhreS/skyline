import pandas as pd

def calculate_overlap(window_start, window_end, row):
    label = row['label']
    start = max(row['label_relative_start_sec'], window_start)
    end = min(row['label_relative_end_sec'], window_end)
    overlap_duration = max(0, end - start)
    return label, overlap_duration

def is_significant_overlap(overlaps, window_size):
    # All overlaps must be more than or equal to 30% of the window size
    for _, overlap in overlaps.items():
        if overlap / window_size < 0.3:
            return False
    return True

def get_unique_labels(overlaps):
    unique_labels = set()
    for label in overlaps.keys():
        for individual_label in label.split(','):
            unique_labels.add(individual_label.strip())
    return ','.join(unique_labels)

def create_windowed_data(df_group, window_size):
    windowed_group_data = []
    df_group.sort_values(by=['label_relative_start_sec'], inplace=True)

    first_start = df_group['label_relative_start_sec'].min()
    last_end = df_group['wav_duration_sec'].iloc[0]

    if first_start > last_end:
        raise ValueError(f"Start time is later than end time, start: {first_start}, end: {last_end}")

    window_start = first_start
    window_end = window_start + window_size
    while window_end <= last_end:
        overlaps = {}
        for _, row in df_group.iterrows():
            label, overlap_duration = calculate_overlap(window_start, window_end, row)
            overlaps[label] = overlaps.get(label, 0) + overlap_duration

        if is_significant_overlap(overlaps, window_size):
            label_str = get_unique_labels(overlaps)
            windowed_group_data.append({
                'wav_blob': df_group['wav_blob'].iloc[0],
                'wav_duration_sec': df_group['wav_duration_sec'].iloc[0],
                'label_duration_sec': window_end - window_start,
                'label_relative_start_sec': window_start,
                'label_relative_end_sec': window_end,
                'label': label_str
            })

        window_start += window_size
        window_end += window_size

    return pd.DataFrame(windowed_group_data)

def window_dataframe_wavs(df, window_size):
    for index, row in df.iterrows():
        if row['label_relative_end_sec'] > row['wav_duration_sec']:
            raise ValueError(f"Label end time is later than wav duration, label end: {row['label_relative_end_sec']}, wav duration: {row['wav_duration_sec']}")

    windowed_df = df.groupby('wav_blob').apply(lambda x: create_windowed_data(x, window_size)).reset_index(drop=True)
    return windowed_df
