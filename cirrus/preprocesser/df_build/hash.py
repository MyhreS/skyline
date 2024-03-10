import pandas as pd
import hashlib


def hash_row(row: pd.Series):
    # Concatenate the string representations of the relevant attributes
    attributes = [
        str(row.get("file_name", "")),
        str(row.get("wav_duration_sec", "")),
        str(row.get("label_duration_sec", "")),
        str(row.get("label_relative_start_sec", "")),
        str(row.get("label_relative_end_sec", "")),
        str(row.get("label", "")),
        str(row.get("sample_rate", "")),
        str(row.get("augmentation", "")),
        str(row.get("audio_format", "")),
        str(row.get("file_type", "")),
    ]
    concatenated = "".join(attributes)
    hash_object = hashlib.sha256(concatenated.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex


def hash(df: pd.DataFrame):
    df["hash"] = df.apply(lambda row: hash_row(row), axis=1)
    return df
