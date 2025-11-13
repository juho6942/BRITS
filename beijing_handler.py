import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from makefiles import load_beijing_data, norm_data, create_fake_missing


def start_index(data, train, seq_len=36):
    """Find valid starting indices for windows based on is_train flag."""
    indices = []
    total_len = data.shape[0]
    for i in range(total_len - seq_len + 1):
        # is_train is the last column in data
        if train:
            # Training: use samples where is_train == 1
            if data[i + seq_len - 1, -1] == 1:
                indices.append(i)
        else:
            # Testing: use samples where is_train == 0
            if data[i + seq_len - 1, -1] == 0:
                indices.append(i)
    return indices


class BeijingData(Dataset):
    """Unified dataset for Beijing air quality data (train and test)."""
    
    def __init__(self, is_train=True, file_path='./Data/beijing_airquality/PRSA_Data_Aotizhongxin_20130301-20170228.csv', missing_rate=0.2):
        super().__init__()
        self.is_train = is_train
        
        # Load and normalize data
        raw_data = load_beijing_data(file_path)
        self.data, self.data_std, self.data_mean, self.missing_mask = norm_data(raw_data)
        
        # Get feature columns
        self.feature_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        
        # Store original normalized values (ground truth for evaluation)
        # These contain the true values before artificial masking
        self.original_values = self.data[self.feature_cols].values.copy()
        
        # Create fake missing data for evaluation
        fake_missing_df, self.fake_missing_mask = create_fake_missing(
            self.data, self.missing_mask, missing_rate=missing_rate
        )
        
        # Extract features with fake missing (NaN where artificially masked)
        # Shape: (time_steps, n_features)
        self.values = fake_missing_df[self.feature_cols].values
        self.original_missing = self.missing_mask[self.feature_cols].values.astype(int)
        self.fake_missing = self.fake_missing_mask[self.feature_cols].values.astype(int)
        self.is_train_flags = self.data['is_train'].values
        
        # Find valid window start indices
        self.indices = start_index(self.data.values, train=is_train)
        
        # Store the absolute starting positions for temporal averaging
        # This allows the evaluate function to map window positions back to absolute time
        self.window_starts = self.indices
    
    def get_normalization_params(self):
        """
        Return normalization parameters for denormalizing predictions.
        
        Returns:
            tuple: (mean, std) as pandas Series with feature names as index
        """
        return self.data_mean, self.data_std
    
    def denormalize(self, normalized_values):
        """
        Denormalize values back to original scale.
        
        Args:
            normalized_values: numpy array of normalized values (z-scores)
        
        Returns:
            numpy array of denormalized values in original units
        """
        # normalized_values = (original - mean) / std
        # original = normalized_values * std + mean
        return normalized_values * self.data_std.values + self.data_mean.values
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        """Return a single window with forward and backward sequences."""
        start_idx = self.indices[index]
        end_idx = start_idx + 36  # SEQ_LEN = 36
        
        # Extract window data (time_steps=36, features=11)
        window_values = self.values[start_idx:end_idx, :]  # (36, 11) - with NaN for masked
        window_ground_truth = self.original_values[start_idx:end_idx, :]  # (36, 11) - original values
        window_original_missing = self.original_missing[start_idx:end_idx, :]
        window_fake_missing = self.fake_missing[start_idx:end_idx, :]
        window_is_train = self.is_train_flags[start_idx:end_idx]
        
        # Combined missing mask (original + fake)
        window_missing = window_original_missing | window_fake_missing
        
        # Create forward sequence
        forward = self._create_sequence(
            window_values, window_ground_truth, window_missing, window_fake_missing, 
            window_is_train, reverse=False
        )
        
        # Create backward sequence (reverse time)
        backward = self._create_sequence(
            window_values, window_ground_truth, window_missing, window_fake_missing,
            window_is_train, reverse=True
        )
        
        # Label: for now, use 0 (you can modify this for classification tasks)
        label = 0
        
        # is_train flag for this window (use the last timestep's flag)
        is_train = window_is_train[-1]
        
        return {
            'forward': forward,
            'backward': backward,
            'label': label,
            'is_train': is_train
        }
    
    def _create_sequence(self, values, ground_truth, missing_mask, eval_mask, is_train_flags, reverse=False):
        """Create a sequence dict with values, masks, deltas, evals, eval_masks.
        
        Args:
            values: normalized values with NaN where masked (for model input)
            ground_truth: original normalized values before masking (for evaluation)
            missing_mask: combined mask (original + fake missing)
            eval_mask: fake missing mask (positions to evaluate)
            is_train_flags: train/test flags
            reverse: whether to reverse the sequence (for backward pass)
        """
        seq_len, n_features = values.shape
        
        if reverse:
            values = values[::-1, :].copy()
            ground_truth = ground_truth[::-1, :].copy()
            missing_mask = missing_mask[::-1, :].copy()
            eval_mask = eval_mask[::-1, :].copy()
        
        # Replace NaN with 0 for model input
        values_filled = np.nan_to_num(values, nan=0.0)
        
        # Compute deltas (time since last observation)
        deltas = np.zeros_like(values)
        for t in range(1, seq_len):
            for f in range(n_features):
                if missing_mask[t, f] == 1:
                    deltas[t, f] = deltas[t-1, f] + 1
                else:
                    deltas[t, f] = 0
        
        # Create sequence as list of timestep dicts
        sequence = []
        for t in range(seq_len):
            timestep = {
                'values': values_filled[t, :].tolist(),
                'masks': (1 - missing_mask[t, :]).tolist(),  # 1 = observed, 0 = missing
                'deltas': deltas[t, :].tolist(),
                'forwards': np.zeros(n_features).tolist(),  # Not used by model, set to zeros
                'evals': ground_truth[t, :].tolist(),  # Ground truth values for evaluation (no NaN)
                'eval_masks': eval_mask[t, :].tolist()  # 1 = evaluate this value
            }
            sequence.append(timestep)
        
        return sequence


# Keep old class names for backward compatibility
class BeijingData_train(BeijingData):
    def __init__(self, file_path='./Data/beijing_airquality/PRSA_Data_Aotizhongxin_20130301-20170228.csv', missing_rate=0.2):
        super().__init__(is_train=True, file_path=file_path, missing_rate=missing_rate)


class BeijingData_test(BeijingData):
    def __init__(self, file_path='./Data/beijing_airquality/PRSA_Data_Aotizhongxin_20130301-20170228.csv', missing_rate=0.2):
        super().__init__(is_train=False, file_path=file_path, missing_rate=missing_rate)


def collate_fn(recs):
    """Collate function matching data_loader.py format exactly."""
    forward = [x['forward'] for x in recs]
    backward = [x['backward'] for x in recs]

    def to_tensor_dict(recs):
        # recs: list of sequences where each sequence is a list of timestep dicts
        # Build tensors matching data_loader.py format
        values = torch.FloatTensor([[x['values'] for x in r] for r in recs])
        masks = torch.FloatTensor([[x['masks'] for x in r] for r in recs])
        deltas = torch.FloatTensor([[x['deltas'] for x in r] for r in recs])
        forwards = torch.FloatTensor([[x['forwards'] for x in r] for r in recs])

        evals = torch.FloatTensor([[x['evals'] for x in r] for r in recs])
        eval_masks = torch.FloatTensor([[x['eval_masks'] for x in r] for r in recs])

        return {
            'values': values,
            'forwards': forwards,
            'masks': masks,
            'deltas': deltas,
            'evals': evals,
            'eval_masks': eval_masks
        }

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    # is_train as CPU tensor (labels not needed for imputation-only task)
    ret_dict['is_train'] = torch.FloatTensor([x['is_train'] for x in recs])

    return ret_dict


def get_loader(batch_size=64, shuffle=True, is_train=True):
    """Get DataLoader for Beijing air quality data."""
    if is_train:
        data_set = BeijingData_train()
    else:
        data_set = BeijingData_test()
    
    data_iter = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for Windows compatibility
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return data_iter
