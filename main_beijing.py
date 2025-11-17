import copy
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    import os
    import glob

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
    patience = 10
    patience_counter = 0
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
            mae, mre = evaluate(model, test_iter, denormalize=True)
            
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch  # ← Remember best epoch
                patience_counter = 0
                old_models = glob.glob(f'best_model_{args.model}_epoch*.pth')
                for old_model in old_models:
                    try:
                        os.remove(old_model)
                        print(f'Deleted old model: {old_model}')
                    except Exception as e:
                        print(f'Could not delete {old_model}: {e}')
                
                # Save new best model with epoch number
                model_path = f'best_model_{args.model}_epoch{epoch}.pth'
                torch.save(model.state_dict(), model_path)
                print(f'✓ New best model saved! MAE: {mae:.6f} → {model_path}\n')
            else:
                patience_counter += 1
                print(f'No improvement ({patience_counter}/{patience})\n')
                
                if patience_counter >= patience:
                    print(f'\n⚠ Early stopping triggered at epoch {epoch}')
                    print(f'Best model was from epoch {best_epoch} with MAE: {best_mae:.6f}')
                    break
    
    # ✅ LOAD BEST MODEL before final evaluation
    print('\n' + '='*60)
    print(f'LOADING BEST MODEL (Epoch {best_epoch}, MAE: {best_mae:.6f})')
    print('='*60)
    model.load_state_dict(torch.load(f'best_model_{args.model}_epoch{best_epoch}.pth'))
    
    # Now these use the BEST model
    print('\n--- Final Evaluation on Best Model ---')
    mae, mre = evaluate(model, test_iter, denormalize=True)
    
    print('\nGenerating predictions file with best model...')
    predict_and_save(model, test_iter, output_file=f'predictions_{args.model}.csv')

def evaluate(model, val_iter, denormalize=True):
    """
    Evaluate model with temporal averaging of imputations.
    
    When the same timestep appears in multiple windows (due to sliding window approach),
    this function averages all imputations for that timestep to get the final prediction.
    This matches the paper's evaluation methodology.
    
    Args:
        model: BRITS model
        val_iter: DataLoader for validation/test set
        denormalize: If True, convert predictions back to original scale
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
    keys = np.array(list(imputation_dict.keys()))  # Shape: [N, 2] where N=6625
    timesteps = keys[:, 0]  # All timesteps
    feature_indices = keys[:, 1].astype(int)  # All feature indices
    
    # Get averaged predictions and ground truths
    evals = np.array([eval_dict[tuple(k)] for k in keys])
    imputations = np.array([np.mean(imputation_dict[tuple(k)]) for k in keys])
    
    print(f"Evaluated {len(imputation_dict)} unique masked values")
    print(f"Average imputations per value: {np.mean([len(v) for v in imputation_dict.values()]):.2f}")
    
    if denormalize:
        mean, std = val_iter.dataset.get_normalization_params()
        feature_cols = val_iter.dataset.feature_cols
        
        # ✅ VECTORIZED: Create arrays of means and stds aligned with feature_indices
        # Build lookup arrays for vectorized operations
        mean_array = np.array([mean[feature_cols[i]] for i in range(len(feature_cols))])
        std_array = np.array([std[feature_cols[i]] for i in range(len(feature_cols))])
        
        # ✅ VECTORIZED DENORMALIZATION: Use feature_indices to select correct mean/std
        evals_denorm = evals * std_array[feature_indices] + mean_array[feature_indices]
        imputations_denorm = imputations * std_array[feature_indices] + mean_array[feature_indices]
        
        # Calculate metrics
        errors_denorm = np.abs(evals_denorm - imputations_denorm)
        mae = errors_denorm.mean()
        mre = errors_denorm.sum() / np.abs(evals_denorm).sum()
        
        print(f'MAE (original scale): {mae:.4f}')
        print(f'MRE (original scale): {mre:.6f}')
        
        # Also compute normalized metrics
        errors_norm = np.abs(evals - imputations)
        mae_norm = errors_norm.mean()
        mre_norm = errors_norm.sum() / np.abs(evals).sum()
        
        print(f'MAE (normalized): {mae_norm:.6f}')
        print(f'MRE (normalized): {mre_norm:.6f}')
        
        # ✅ BONUS: Per-feature statistics using vectorized operations
        print('\n=== Summary by Feature ===')
        for f_idx in range(len(feature_cols)):
            mask = feature_indices == f_idx
            if mask.sum() > 0:
                feat_mae = errors_denorm[mask].mean()
                feat_count = mask.sum()
                print(f'{feature_cols[f_idx]:8s}: MAE = {feat_mae:.4f} (n={feat_count})')
        
        return mae, mre
    
    else:
        # No denormalization
        errors = np.abs(evals - imputations)
        mae = errors.mean()
        mre = errors.sum() / np.abs(evals).sum()
        
        print(f'MAE: {mae:.6f}')
        print(f'MRE: {mre:.6f}')
        
        return mae, mre

def predict_and_save(model, val_iter, output_file='predictions.csv'):
    """
    Generate predictions and save to CSV with denormalized values.
    
    Args:
        model: Trained BRITS model
        val_iter: DataLoader
        output_file: Path to save predictions
    """
    model.eval()
    
    # Store all predictions with metadata
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, None)

            eval_masks = ret['eval_masks'].data.cpu().numpy()
            eval_ = ret['evals'].data.cpu().numpy()
            imputation = ret['imputations'].data.cpu().numpy()
            
            batch_size = eval_masks.shape[0]
            seq_len = eval_masks.shape[1]
            n_features = eval_masks.shape[2]
            
            for b in range(batch_size):
                sample_idx = batch_idx * val_iter.batch_size + b
                if sample_idx >= len(val_iter.dataset):
                    break
                
                window_start = val_iter.dataset.window_starts[sample_idx]
                
                for t in range(seq_len):
                    absolute_time = window_start + t
                    
                    for f in range(n_features):
                        if eval_masks[b, t, f] == 1:  # Artificially masked
                            results.append({
                                'timestep': absolute_time,
                                'feature_idx': f,
                                'feature_name': val_iter.dataset.feature_cols[f],
                                'true_normalized': eval_[b, t, f],
                                'pred_normalized': imputation[b, t, f]
                            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Denormalize values
    mean, std = val_iter.dataset.get_normalization_params()
    
    for idx, row in df.iterrows():
        feat_name = row['feature_name']
        df.at[idx, 'true_value'] = row['true_normalized'] * std[feat_name] + mean[feat_name]
        df.at[idx, 'pred_value'] = row['pred_normalized'] * std[feat_name] + mean[feat_name]
        df.at[idx, 'absolute_error'] = abs(df.at[idx, 'true_value'] - df.at[idx, 'pred_value'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f'\nPredictions saved to {output_file}')
    print(f'Columns: {list(df.columns)}')
    
    # Print summary statistics by feature
    print('\n=== Summary by Feature ===')
    for feat in val_iter.dataset.feature_cols:
        feat_df = df[df['feature_name'] == feat]
        mae = feat_df['absolute_error'].mean()
        print(f'{feat:8s}: MAE = {mae:.4f} (n={len(feat_df)})')
    
    return df

def run():
    if torch is None:
        raise RuntimeError('PyTorch is not installed or could not be imported. Please install torch to run this script.')

    # Create model in imputation-only mode (Beijing has 11 features, 36 timesteps)
    model = getattr(models, args.model).Model(imputation_only=True, features=11, seq_len=36)

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
