# Evaluation Methodology - Temporal Averaging

## Overview

This implementation follows the paper's evaluation approach for time-series imputation, where **multiple imputations for the same timestep are averaged** to produce the final prediction.

## Why Temporal Averaging?

When using a sliding window approach with 36-step windows, the same timestep can appear in multiple windows:

```
Example: Timestep 100 can appear in:
- Window 1: timesteps 65-100  → produces imputation_1
- Window 2: timesteps 66-101  → produces imputation_2
- Window 3: timesteps 67-102  → produces imputation_3
...
- Window 36: timesteps 100-135 → produces imputation_36

Final imputation = mean(imputation_1, ..., imputation_36)
```

## Implementation Details

### Key Function: `evaluate(model, val_iter)`

**What it does:**
1. Collects all imputations for each unique (timestep, feature) pair
2. Averages multiple predictions for the same timestep
3. Computes MAE/MRE on these averaged imputations

**Data Structure:**
```python
imputation_dict = {
    (absolute_time_index, feature_index): [imp1, imp2, imp3, ...],
    ...
}
```

**Process:**
```
For each batch:
  For each sample in batch:
    window_start = dataset.window_starts[sample_idx]
    
    For each timestep t in [0, 35]:
      absolute_time = window_start + t
      
      For each feature f in [0, 10]:
        if this value was artificially masked:
          store imputation at key (absolute_time, f)

After all batches:
  For each unique (time, feature) key:
    final_imputation = mean(all stored imputations)
    compute |true_value - final_imputation|
```

## Benefits

1. **Variance Reduction**: Averaging reduces random noise in predictions
2. **Paper Compliance**: Matches the evaluation methodology in the BRITS paper
3. **Real-World Accuracy**: Better reflects how the model would be deployed
4. **Fair Comparison**: Ensures each unique timestep is counted once

## Output Format

```
--- Evaluation at Epoch 10 ---
Evaluated 1234 unique masked values
Average imputations per value: 18.50
MAE: 0.345678
MRE: 0.234567
```

**Interpretation:**
- `Evaluated X unique masked values`: Total number of (timestep, feature) pairs that were artificially masked
- `Average imputations per value`: How many windows contained each timestep (typically ~18-36 depending on data)
- `MAE`: Mean Absolute Error on averaged imputations
- `MRE`: Mean Relative Error on averaged imputations

## Train/Test Split

Following the paper:
- **Train months**: 1, 2, 4, 5, 7, 8, 10, 11 (8 months)
- **Test months**: 3, 6, 9, 12 (4 months)
- **Window size**: 36 consecutive timesteps
- **Sampling**: Random during training (`shuffle=True`)

## Usage

```bash
# Train BRITS model on Beijing air quality data
python main_beijing.py --epochs 1000 --batch_size 32 --model brits

# The script will:
# 1. Train on months 1,2,4,5,7,8,10,11
# 2. Evaluate on months 3,6,9,12 with temporal averaging
# 3. Save best model based on test MAE
```

## Performance Expectations

Based on the paper:
- **MAE**: Lower is better (typically 0.2-0.4 for normalized data)
- **MRE**: Lower is better (typically 0.15-0.35)
- **Training time**: ~1-2 minutes per epoch (depending on GPU)
- **Convergence**: Usually within 100-200 epochs

## Comparison to Alternative Methods

| Method | Pros | Cons |
|--------|------|------|
| **Independent Windows** | Faster evaluation | Inflated sample count, doesn't reflect deployment |
| **Temporal Averaging** (Ours) | Paper-compliant, fair evaluation | Slightly slower |
| **Ensemble Prediction** | Best for deployment | Computational overhead |

---

*Last updated: 2025-11-13*
