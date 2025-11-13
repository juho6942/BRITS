import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


def load_beijing_data(file_path):
    data = pd.read_csv(file_path)
    
    train_months = [1,2,4,5,7,8,10,11]
    data['is_train'] = data['month'].apply(lambda x: 1 if x in train_months else 0)
    data = data[data['year'] == 2015] #change this to select which year to use
    return data

def norm_data(data):
    feature_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    missing_vals = ['NA', 'NaN', 'nan', '--', '']
    missing_mask = data[feature_cols].isin(missing_vals)
    data[feature_cols] = data[feature_cols].replace(missing_vals, np.nan)
    
    data[feature_cols] = data[feature_cols].astype(float)
    data_mean = data[feature_cols].mean()
    data_std = data[feature_cols].std()

    data[feature_cols] = (data[feature_cols] - data_mean) / data_std
    data[missing_mask] = 0
    return data,data_std, data_mean, missing_mask

def create_fake_missing(data, missing_mask, missing_rate=0.11):
    
    feature_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    featuredata = data[feature_cols].values
    # Generate a random mask for new missing values
    new_mask = np.random.choice([0, 1], size=featuredata.shape, p=[1-missing_rate, missing_rate])
    # Ensure we do not overwrite existing missing values
    new_mask = new_mask & ~missing_mask.values
    featuredata[new_mask == 1] = np.nan
    fake_missing = data.copy()
    fake_missing[feature_cols] = featuredata
    return fake_missing, pd.DataFrame(new_mask,columns=feature_cols)


data = load_beijing_data('./Data/beijing_airquality/PRSA_Data_Aotizhongxin_20130301-20170228.csv')
data_normed,data_std, data_mean, original_missing_mask = norm_data(data)
fake_missing_data, fake_missing_mask = create_fake_missing(data, original_missing_mask, missing_rate=0.2)

print("Data loading and preprocessing complete.")