import os
import time

try:
    import ujson as json
except Exception:
    import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self):
        super().__init__()
        # use context manager to open file
        with open('./json/json', 'r') as f:
            self.content = f.readlines()

        indices = np.arange(len(self.content))
        val_indices = indices[350:]

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

def collate_fn(recs):
    forward = [x['forward'] for x in recs]
    backward = [x['backward'] for x in recs]

    def to_tensor_dict(recs):
        # recs: list of sequences where each element is a dict-like with lists
        # Build CPU tensors here. DataLoader will pin memory if pin_memory=True.
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

    # labels and is_train as CPU tensors; training loop will move them to device
    ret_dict['labels'] = torch.FloatTensor([x['label'] for x in recs])
    ret_dict['is_train'] = torch.FloatTensor([x['is_train'] for x in recs])

    return ret_dict

def get_loader(batch_size = 64, shuffle = True, is_train=None):
    """
    Get DataLoader for train or test set.
    
    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        is_train: If True, only train data. If False, only test data. If None, all data.
    """
    data_set = MySet()
    
    # Filter indices based on is_train flag
    if is_train is True:
        # Only training samples (indices 0-349)
        indices = [i for i in range(len(data_set)) if i not in data_set.val_indices]
    elif is_train is False:
        # Only test samples (indices 350-399)
        indices = list(data_set.val_indices)
    else:
        # All samples (original behavior)
        indices = list(range(len(data_set)))
    
    # Create sampler for subset
    from torch.utils.data import SubsetRandomSampler, SequentialSampler
    if shuffle and is_train is not None:
        sampler = SubsetRandomSampler(indices)
    elif is_train is not None:
        sampler = SequentialSampler(indices)
    else:
        sampler = None
    
    data_iter = DataLoader(
        dataset = data_set,
        batch_size = batch_size,
        num_workers = 4,
        sampler = sampler if is_train is not None else None,
        shuffle = shuffle if is_train is None else False,
        pin_memory = True,
        collate_fn = collate_fn
    )

    return data_iter
