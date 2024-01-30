import pandas as pd

def train_val_test_split(df, train_percent, test_percent, validation_percent):
    # Create split column and fill with 'train'
    df['split'] = 'train'
    smallest_label_size = df.groupby('label').size().min()
    test_amount = int(smallest_label_size * (test_percent / 100))

    for label, group in df.groupby('label'):
        validation_amount_label = int(len(group) * (validation_percent / 100))
        df.loc[group.index[:test_amount], 'split'] = 'test'
        df.loc[group.index[-validation_amount_label:], 'split'] = 'validation'

    return df






    
