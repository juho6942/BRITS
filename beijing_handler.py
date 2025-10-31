import pandas as pd
import numpy as np


def load_beijing_data(file_path):
    data = pd.read_csv(file_path)
    missing_vals = ['NA', 'NaN', 'nan', '--', '']
    missing_mask = data.isin(missing_vals)

    data[missing_mask] = 0
    train_months = [1,2,4,5,7,8,10,11]
    data['is_train'] = data['month'].apply(lambda x: 1 if x in train_months else 0)

    return norm_data
def norm_data(data):
    feature_cols = data.columns.difference(['year', 'month', 'day', 'hour', 'is_train'])
    data_mean = data[feature_cols].mean()
    data_std = data[feature_cols].std()

    data[feature_cols] = (data[feature_cols] - data_mean) / data_std

    return data
def start_index(data, seq_len = 36):
    indices = []
    total_len = data.shape[0]
    for i in range(total_len - seq_len + 1):
        if data[i + seq_len - 1, -1] == 1:  # Check is_train flag at the end of the sequence
            indices.append(i)
    return indices
