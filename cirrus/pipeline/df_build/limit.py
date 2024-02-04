import pandas as pd


def limit(df: pd.DataFrame, limit: int): # TODO: Sampe on duration. And sample on duration sec
    assert len(df) > 0, "Dataframe is empty"
    assert 'label' in df.columns, "Dataframe does not contain 'label' column"
    assert 'split' in df.columns, "Dataframe does not contain 'split' column"
    assert 'augmentation' in df.columns, "Dataframe does not contain 'augmentation' column"

    to_keep = []
    for index, group in df.groupby('split'):
        if len(group) < limit:
            to_keep.append(group)
            continue
        group = group.sample(limit, random_state=1)
        to_keep.append(group)
    return pd.concat(to_keep)


        


