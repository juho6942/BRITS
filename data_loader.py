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
        val_indices = np.random.choice(indices, len(self.content) // 5)

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

def get_loader(batch_size = 64, shuffle = True):
    data_set = MySet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
