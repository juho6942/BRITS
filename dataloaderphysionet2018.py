import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler




class PhysioNetDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        self.is_train = is_train
        self.data = pd.read_csv(csv_file)
        self.realmissing = pd.read_csv('Data/combined_missing_mask.csv')
        self.evaluation = pd.read_csv('Data/combined_eval_mask.csv')
        self.features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
        
        self.mean,self.std = get_meanstd(self.data[self.features].values, 1 - (self.realmissing[self.features].values + self.evaluation[self.features].values))
        # Process PatientID
        for df in [self.data, self.realmissing, self.evaluation]:
            df['PatientID'] = df['PatientID'].astype(str).str.replace('p', '').astype(int)
            
        if is_train:
            self.data = self.data[self.data['PatientID'] < 4000]
            self.realmissing = self.realmissing[self.realmissing['PatientID'] < 4000]
            self.evaluation = self.evaluation[self.evaluation['PatientID'] < 4000]
        else:
            self.data = self.data[self.data['PatientID'] >= 4000]
            self.realmissing = self.realmissing[self.realmissing['PatientID'] >= 4000]
            self.evaluation = self.evaluation[self.evaluation['PatientID'] >= 4000]

        self.standardized = self.data.copy()
        self.standardized[self.features] = (self.standardized[self.features] - self.mean) / self.std
        # Group by PatientID for fast access
        self.data_grouped = self.standardized.groupby('PatientID')
        self.realmissing_grouped = self.realmissing.groupby('PatientID')
        self.evaluation_grouped = self.evaluation.groupby('PatientID')


        self.patient_ids = list(self.data_grouped.groups.keys())
        
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        sample = self.data_grouped.get_group(patient_id)
        realmissing = self.realmissing_grouped.get_group(patient_id)
        evaluation = self.evaluation_grouped.get_group(patient_id)
        
        # Extract label (assuming SepsisLabel is constant or we take max)
        label = sample['SepsisLabel'].max()
        
        values = sample[self.features].values
        rm = realmissing[self.features].values
        ev = evaluation[self.features].values
        
        # Masks: 1 if observed, 0 if missing (real missing or eval mask)
        masks = 1 - (rm + ev)
        
        # Deltas
        deltas = self._create_deltas(values, masks)
        
        # Backward
        values_b = values[::-1]
        masks_b = masks[::-1]
        deltas_b = self._create_deltas(values_b, masks_b)
        
        # Evals
        evals = sample[self.features].values
        eval_masks = ev
        
        return {
            'forward': {
                'values': values,
                'masks': masks,
                'deltas': deltas,
                'evals': evals,
                'eval_masks': eval_masks
            },
            'backward': {
                'values': values_b,
                'masks': masks_b,
                'deltas': deltas_b,
                'evals': evals[::-1],
                'eval_masks': eval_masks[::-1]
            },
            'label': label,
            'is_train': 1 if self.is_train else 0
        }
    
    def _create_deltas(self, values, masks):
        """Compute time deltas for each feature."""
        deltas = np.zeros_like(values)
        n_timesteps, n_features = values.shape
        
        for feature_idx in range(n_features):
            last_observed = -1
            for t in range(n_timesteps):
                if masks[t, feature_idx] == 1:
                    if last_observed == -1:
                        deltas[t, feature_idx] = 0
                    else:
                        deltas[t, feature_idx] = t - last_observed
                    last_observed = t
                else:
                    if last_observed == -1:
                        deltas[t, feature_idx] = 0
                    else:
                        deltas[t, feature_idx] = t - last_observed
        return deltas
    def get_meanstd(self):
        return self.mean, self.std

def get_meanstd(data, masks):
    """Normalize data to zero mean and unit variance, ignoring missing values."""
    # data contains 0 for missing values. masks contains 0 for missing values.
    # We need to convert masked values to NaN so nanmean/nanstd ignores them.
    
    data_masked = data.copy()
    data_masked[masks == 0] = np.nan
    
    # Calculate mean and std per feature (axis=0)
    # Use a large epsilon to avoid issues with empty slices if a feature is completely missing
    with np.errstate(all='ignore'):
        means = np.nanmean(data_masked, axis=0)
        stds = np.nanstd(data_masked, axis=0)
    
    # Handle features that are completely missing (NaN mean/std)
    means = np.nan_to_num(means, nan=0.0)
    stds = np.nan_to_num(stds, nan=1.0)

    # Avoid division by zero
    if np.isscalar(stds):
        if stds == 0:
            stds = 1.0
    else:
        stds[stds == 0] = 1.0
    
    return means, stds

def collate_fn(recs):
    def to_tensor_dict(recs, direction):
        values = torch.FloatTensor(np.array([r[direction]['values'] for r in recs]))
        masks = torch.FloatTensor(np.array([r[direction]['masks'] for r in recs]))
        deltas = torch.FloatTensor(np.array([r[direction]['deltas'] for r in recs]))
        evals = torch.FloatTensor(np.array([r[direction]['evals'] for r in recs]))
        eval_masks = torch.FloatTensor(np.array([r[direction]['eval_masks'] for r in recs]))
        
        return {
            'values': values,
            'masks': masks,
            'deltas': deltas,
            'evals': evals,
            'eval_masks': eval_masks
        }

    ret_dict = {
        'forward': to_tensor_dict(recs, 'forward'),
        'backward': to_tensor_dict(recs, 'backward'),
        'labels': torch.FloatTensor([r['label'] for r in recs]),
        'is_train': torch.FloatTensor([r['is_train'] for r in recs])
    }
    return ret_dict

def get_loader(batch_size = 64, shuffle = True, is_train=True):
    """
    Get DataLoader for train or test set.
    
    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        is_train: If True, only train data. If False, only test data. If None, all data.
    """
    dataset = PhysioNetDataset('Data/combined_patient_data.csv', is_train=is_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader