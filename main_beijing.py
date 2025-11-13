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
import beijing_handler
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

    
    train_iter = beijing_handler.get_loader(batch_size=args.batch_size, shuffle=True, is_train=True)
    test_iter = beijing_handler.get_loader(batch_size=args.batch_size, shuffle=False, is_train=False)
    
    print(f"Training windows: {len(train_iter.dataset)}")
    print(f"Testing windows: {len(test_iter.dataset)}\n")
    
    best_mae = float('inf')

    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0

        
        for idx, data in enumerate(train_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)

            loss_val = None
            if hasattr(ret['loss'], 'item'):
                loss_val = ret['loss'].item()
            else:
                try:
                    loss_val = ret['loss'].data[0]
                except Exception:
                    loss_val = float(ret['loss'])

            run_loss += loss_val

            print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(train_iter), run_loss / (idx + 1.0)), end='')
        
        avg_loss = run_loss / len(train_iter)
        print(f'\n[Epoch {epoch}] Average Training Loss: {avg_loss:.6f}')
        
        if torch.cuda.is_available():
            print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
         
        if epoch % 1 == 0:
            print(f'\n--- Evaluation at Epoch {epoch} ---')
            mae, mre = evaluate(model, test_iter)
            
            # Save best model
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), f'best_model_{args.model}_epoch{epoch}.pth')
                print(f'✓ New best model saved! MAE: {mae:.6f}\n')
            else:
                print()  # Empty line for readability

def evaluate(model, val_iter):
    """
    Evaluate model with temporal averaging of imputations.
    
    When the same timestep appears in multiple windows (due to sliding window approach),
    this function averages all imputations for that timestep to get the final prediction.
    This matches the paper's evaluation methodology.
    """
    model.eval()

    # Dictionary to store multiple imputations for the same timestep
    # Key: (absolute_time_index, feature_index)
    # Value: list of imputed values from different windows
    imputation_dict = {}
    eval_dict = {}
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, None)

            eval_masks = ret['eval_masks'].data.cpu().numpy()  # [batch, seq_len, features]
            eval_ = ret['evals'].data.cpu().numpy()
            imputation = ret['imputations'].data.cpu().numpy()
            
            batch_size = eval_masks.shape[0]
            seq_len = eval_masks.shape[1]
            n_features = eval_masks.shape[2]
            
            # For each sample in the batch
            for b in range(batch_size):
                # Calculate the global sample index
                sample_idx = batch_idx * val_iter.batch_size + b
                if sample_idx >= len(val_iter.dataset):
                    break
                
                # Get the absolute starting position of this window in the full time series
                window_start = val_iter.dataset.window_starts[sample_idx]
                
                # For each timestep in the 36-step window
                for t in range(seq_len):
                    absolute_time = window_start + t
                    
                    # For each feature (PM2.5, PM10, SO2, etc.)
                    for f in range(n_features):
                        if eval_masks[b, t, f] == 1:  # This value was artificially masked
                            key = (absolute_time, f)
                            
                            # Store the imputation
                            if key not in imputation_dict:
                                imputation_dict[key] = []
                                eval_dict[key] = eval_[b, t, f]  # Ground truth (same for all windows)
                            
                            imputation_dict[key].append(imputation[b, t, f])
    
    # Average multiple imputations for each unique timestep
    evals = []
    imputations = []
    
    for key in sorted(imputation_dict.keys()):  # Sort for consistent ordering
        evals.append(eval_dict[key])
        # Average all imputations from different windows containing this timestep
        imputations.append(np.mean(imputation_dict[key]))
    
    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    
    # Calculate metrics
    mae = np.abs(evals - imputations).mean()
    mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()
    
    print(f'Evaluated {len(evals)} unique masked values')
    print(f'Average imputations per value: {np.mean([len(v) for v in imputation_dict.values()]):.2f}')
    print(f'MAE: {mae:.6f}')
    print(f'MRE: {mre:.6f}')
    
    return mae, mre

def run():
    if torch is None:
        raise RuntimeError('PyTorch is not installed or could not be imported. Please install torch to run this script.')

    model = getattr(models, args.model).Model()

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
