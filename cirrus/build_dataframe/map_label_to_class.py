import pandas as pd
from typing import Dict


def _map_label_to_class(row: pd.Series, label_to_class_map: Dict):
    for class_name, labels in label_to_class_map.items():
        if row['label'] in labels:
            row['class'] = class_name
            return row
    raise Exception(f"Label {row['label']} not found in label_to_class_map {label_to_class_map}")

def map_label_to_class(df: pd.DataFrame, label_to_class_map: Dict):
    assert 'label' in df.columns, "df must contain column 'label'"
    assert len(label_to_class_map) > 0, "label_to_class_map must be non-empty"    
    df['class'] = None
    mapped_df = df.apply(lambda row: _map_label_to_class(row, label_to_class_map), axis=1)
    # Check that no class is None
    assert mapped_df['class'].isnull().sum() == 0
    return mapped_df
    