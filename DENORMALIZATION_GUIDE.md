# Denormalization Guide for Beijing Air Quality Imputation

## Overview

The model trains on **normalized data (z-scores)** but evaluates and reports metrics in **original scale** (µg/m³ for PM2.5, °C for temperature, etc.). This makes results interpretable and comparable to real-world measurements.

---

## Why Denormalize?

### Training on Normalized Data
```python
# During preprocessing in norm_data():
normalized = (original - mean) / std

# Example:
PM2.5_original = 85.3 µg/m³
PM2.5_mean = 89.5 µg/m³
PM2.5_std = 75.2 µg/m³

PM2.5_normalized = (85.3 - 89.5) / 75.2 = -0.056
```

**Why normalize for training?**
- Puts all features on same scale (PM2.5 ~100, TEMP ~10, PRES ~1000)
- Improves gradient descent convergence
- Prevents features with large values from dominating loss

### Evaluation on Original Scale
```python
# During evaluation:
original = normalized * std + mean

PM2.5_predicted_norm = -0.045
PM2.5_predicted_original = (-0.045 * 75.2) + 89.5 = 86.1 µg/m³
```

**Why denormalize for evaluation?**
- **Interpretable**: "MAE = 12.5 µg/m³" is meaningful to domain experts
- **Comparable**: Can compare with other papers using original units
- **Real-world**: Shows actual prediction accuracy in operational settings

---

## Implementation

### 1. Store Normalization Parameters (in `beijing_handler.py`)

```python
class BeijingData(Dataset):
    def __init__(self, ...):
        # Store mean and std from normalization
        self.data, self.data_std, self.data_mean, self.missing_mask = norm_data(raw_data)
    
    def get_normalization_params(self):
        """Return (mean, std) for denormalization."""
        return self.data_mean, self.data_std
    
    def denormalize(self, normalized_values):
        """Convert z-scores back to original scale."""
        return normalized_values * self.data_std.values + self.data_mean.values
```

**Stored Parameters:**
```
Feature   | Mean (µ)  | Std (σ)
----------|-----------|----------
PM2.5     | 89.5      | 75.2
PM10      | 108.2     | 87.3
SO2       | 25.3      | 34.1
NO2       | 50.7      | 30.8
CO        | 1150.4    | 815.2
O3        | 56.8      | 51.3
TEMP      | 12.3      | 11.5
PRES      | 1014.2    | 10.8
DEWP      | 1.8       | 13.1
RAIN      | 0.09      | 0.52
WSPM      | 2.3       | 1.4
```

### 2. Evaluate with Denormalization (in `main_beijing.py`)

```python
def evaluate(model, val_iter, denormalize=True):
    # ... collect normalized predictions ...
    
    if denormalize:
        # Get normalization params
        mean, std = val_iter.dataset.get_normalization_params()
        
        # Denormalize: original = normalized * std + mean
        evals_denorm = evals_2d * std.values + mean.values
        imputations_denorm = imputations_2d * std.values + mean.values
        
        # Calculate metrics on original scale
        mae = np.abs(evals_denorm - imputations_denorm).mean()
        mre = np.abs(evals_denorm - imputations_denorm).sum() / np.abs(evals_denorm).sum()
        
        print(f'MAE (original scale): {mae:.4f}')  # e.g., "MAE: 12.34 µg/m³"
        print(f'MRE (original scale): {mre:.6f}')
```

### 3. Save Denormalized Predictions

```python
def predict_and_save(model, val_iter, output_file='predictions.csv'):
    # ... collect predictions ...
    
    # Denormalize each prediction
    mean, std = val_iter.dataset.get_normalization_params()
    
    for idx, row in df.iterrows():
        feat_name = row['feature_name']
        df.at[idx, 'true_value'] = row['true_normalized'] * std[feat_name] + mean[feat_name]
        df.at[idx, 'pred_value'] = row['pred_normalized'] * std[feat_name] + mean[feat_name]
    
    df.to_csv(output_file, index=False)
```

**Output CSV Format:**
```csv
timestep,feature_idx,feature_name,true_normalized,pred_normalized,true_value,pred_value,absolute_error
1000,0,PM2.5,-0.056,-0.045,85.30,86.12,0.82
1000,1,PM10,0.123,0.145,118.94,121.87,2.93
...
```

---

## Expected Output

### During Training

```
Training windows: 5234
Testing windows: 2617

Progress epoch 0, 100.00%, average loss 0.4523
[Epoch 0] Average Training Loss: 0.452300

--- Evaluation at Epoch 0 ---
Evaluated 8456 unique masked values
Average imputations per value: 18.50
MAE (original scale): 15.2340      ← Real units (µg/m³, °C, hPa, etc.)
MRE (original scale): 0.234567
MAE (normalized): 0.202345         ← Z-scores (for reference)
MRE (normalized): 0.234567
✓ New best model saved! MAE: 15.234000

...

--- Evaluation at Epoch 50 ---
Evaluated 8456 unique masked values
Average imputations per value: 18.50
MAE (original scale): 12.5678      ← Improved!
MRE (original scale): 0.198234
MAE (normalized): 0.167234
MRE (normalized): 0.198234
✓ New best model saved! MAE: 12.567800
```

### Final Predictions CSV

```
Predictions saved to predictions_brits.csv
Columns: ['timestep', 'feature_idx', 'feature_name', 'true_normalized', 
          'pred_normalized', 'true_value', 'pred_value', 'absolute_error']

=== Summary by Feature ===
PM2.5   : MAE = 12.5678 (n=768)   ← µg/m³
PM10    : MAE = 18.3421 (n=768)   ← µg/m³
SO2     : MAE = 8.2341 (n=768)    ← µg/m³
NO2     : MAE = 10.4567 (n=768)   ← µg/m³
CO      : MAE = 201.234 (n=768)   ← µg/m³
O3      : MAE = 15.6789 (n=768)   ← µg/m³
TEMP    : MAE = 2.3456 (n=768)    ← °C
PRES    : MAE = 3.4567 (n=768)    ← hPa
DEWP    : MAE = 2.8901 (n=768)    ← °C
RAIN    : MAE = 0.1234 (n=768)    ← mm
WSPM    : MAE = 0.5678 (n=768)    ← m/s
```

---

## Comparison: Normalized vs Denormalized Metrics

| Metric Type | When to Use | Example Value | Interpretation |
|-------------|-------------|---------------|----------------|
| **Normalized MAE** | Compare to paper (if they use z-scores) | 0.167 | "0.167 standard deviations off" |
| **Denormalized MAE** | Real-world interpretation, domain experts | 12.57 µg/m³ | "12.57 µg/m³ average error for PM2.5" |
| **MRE** | Relative error (scale-independent) | 0.198 | "19.8% relative error" |

---

## Best Practices

### 1. **Report Both Metrics**
```python
# Always show both for transparency
print(f'MAE (original scale): {mae_denorm:.4f}')
print(f'MAE (normalized): {mae_norm:.6f}')
```

### 2. **Use Original Scale for Papers**
- Most air quality papers report in µg/m³
- Makes results comparable across studies
- Domain experts can assess practical utility

### 3. **Feature-Specific Analysis**
```python
# Different features have different scales
PM2.5: MAE = 12.5 µg/m³   (good if typical range is 0-200)
TEMP:  MAE = 2.3 °C       (good if typical range is -10 to 35)
PRES:  MAE = 3.4 hPa      (good if typical range is 980-1040)
```

### 4. **Verify Denormalization**
```python
# Sanity check: denormalized values should be in reasonable range
assert 0 <= pm25_pred <= 500, "PM2.5 out of physical range!"
assert -50 <= temp_pred <= 50, "Temperature out of range!"
```

---

## Mathematical Formula

```
Training:
  X_norm = (X - μ) / σ
  Model learns: f(X_norm) → Y_norm

Evaluation:
  Y_pred_norm = f(X_norm)
  Y_pred_original = Y_pred_norm × σ + μ
  
  MAE = mean(|Y_true_original - Y_pred_original|)
```

---

## Usage

```bash
# Run training with denormalized evaluation (default)
python main_beijing.py --epochs 100 --batch_size 32 --model brits

# Output files:
# - best_model_brits_epochXX.pth      (best model checkpoint)
# - predictions_brits.csv              (denormalized predictions)
```

The model will:
1. ✅ Train on normalized data (z-scores)
2. ✅ Evaluate on original scale (interpretable units)
3. ✅ Save predictions in both normalized and denormalized form
4. ✅ Report MAE in real-world units (µg/m³, °C, hPa, etc.)

---

*Last updated: 2025-11-13*
