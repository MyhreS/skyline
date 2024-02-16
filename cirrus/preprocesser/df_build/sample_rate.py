import pandas as pd

def sample_rate(df: pd.DataFrame, sample_rate: int = 44100):
    """
    Set the sample rate for the data
    """
    assert df is not None, "df must be non-empty"
    df["sample_rate"] = sample_rate
    return df