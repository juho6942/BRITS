import copy
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
except Exception:
    # Torch may not be installed in the editor environment; allow file to be read/checked.
    torch = None
    nn = None
    F = None
    optim = None
    StepLR = None

import numpy as np

import time
import utils
import models
import argparse
import data_loader
import pandas as pd
try:
    import ujson as json
except Exception:
    import json

from sklearn import metrics

try:
    from ipdb import set_trace
except Exception:
    try:
        from pdb import set_trace
    except Exception:
        def set_trace():
            return None

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 1000)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--model', type = str)
args = parser.parse_args()

def train(model):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    data_iter = data_loader.get_loader(batch_size = args.batch_size)

    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)

            # get loss value (PyTorch 0.x / 1.x compatibility)
            loss_val = None
            if hasattr(ret['loss'], 'item'):
                loss_val = ret['loss'].item()
            else:
                # older versions
                try:
                    loss_val = ret['loss'].data[0]
                except Exception:
                    loss_val = float(ret['loss'])

            run_loss += loss_val

            # Python3 print
            print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)), end='')
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        if epoch % 1 == 0:
            evaluate(model, data_iter)

def evaluate(model, val_iter):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)

    print('AUC {}'.format(metrics.roc_auc_score(labels, preds)))

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    print('MAE', np.abs(evals - imputations).mean())
    print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())

def run():
    if torch is None:
        raise RuntimeError('PyTorch is not installed or could not be imported. Please install torch to run this script.')

    model = getattr(models, args.model).Model()

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
