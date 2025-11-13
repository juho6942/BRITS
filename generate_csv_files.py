
import sys
import os


from makefiles import load_beijing_data, norm_data, create_fake_missing
import pandas as pd

# Load the data and masks
data = load_beijing_data('./Data/beijing_airquality/PRSA_Data_Aotizhongxin_20130301-20170228.csv')
data_normed,data_std, data_mean, original_missing_mask= norm_data(data)
fake_missing_data, fake_missing_mask = create_fake_missing(data, original_missing_mask, missing_rate=0.2)

# Save the train data to a CSV file
data_normed.to_csv('./Data/data_for_team/data.csv', index=False)
print("Train data saved to './Data/data_for_team/data.csv'.")

# Save the fake missing mask to a CSV file
fake_missing_mask.to_csv('./Data/data_for_team/fake_missing_mask.csv', index=False)
print("Fake missing mask saved to './Data/data_for_team/fake_missing_mask.csv'.")

# Save the fake missing data to a CSV file
fake_missing_data.to_csv('./Data/data_for_team/fake_missing_data.csv', index=False)
print("Fake missing data saved to './Data/data_for_team/fake_missing_data.csv'.")