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
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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
#parser.add_argument('--impute_only', type = bool, default = False)
args = parser.parse_args()

def train(model):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # Separate train and test loaders
    train_iter = data_loader.get_loader(batch_size = args.batch_size, shuffle=True, is_train=True)
    test_iter = data_loader.get_loader(batch_size = args.batch_size, shuffle=False, is_train=False)
    
    #print(f"Training samples: {len(train_iter.dataset) - len(train_iter.dataset.val_indices)}")  # 350
    #print(f"Test samples: {len(train_iter.dataset.val_indices)}")  # 50

    train_loss_history = []
    eval_mae_history = []

    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0

        # Only iterate over TRAINING data
        for idx, data in enumerate(train_iter):
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
            print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(train_iter), run_loss / (idx + 1.0)), end='')
        
        avg_loss = run_loss / len(train_iter)
        train_loss_history.append(avg_loss)

        print(f"\nAllocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        if epoch % 1 == 0:
            # Evaluate on TEST data only
            mae = evaluate(model, test_iter)
            eval_mae_history.append(mae)

    # Plotting
    if plt:
        try:
            plt.figure(figsize=(10, 6))
            # plt.plot(train_loss_history, label='Train Loss')
            plt.plot(eval_mae_history, label='Eval MAE')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Evaluation MAE')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'loss_graph_{args.model}.png')
            print(f'Graph saved to loss_graph_{args.model}.png')
        except Exception as e:
            print(f'Could not generate graph: {e}')

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

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # All samples in test_iter are test samples, no filtering needed
        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)

    print('AUC {}'.format(metrics.roc_auc_score(labels, preds)))

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    print('MAE', np.abs(evals - imputations).mean())
    print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())
    return np.abs(evals - imputations).mean()

def run():
    if torch is None:
        raise RuntimeError('PyTorch is not installed or could not be imported. Please install torch to run this script.')

    # PhysioNet data has 35 features and 49 timesteps
    model = getattr(models, args.model).Model(imputation_only=True, features=35, seq_len=49)

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
