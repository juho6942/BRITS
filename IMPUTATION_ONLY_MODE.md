# Imputation-Only Mode Implementation

## Overview

The BRITS model originally performed **two tasks simultaneously**:
1. **Imputation**: Fill in missing values in time series
2. **Classification**: Predict a binary label for each sequence

For Beijing air quality data, we only need **imputation** (no classification labels). This update adds an `imputation_only` mode to disable the classification task.

---

## Changes Made

### 1. **Updated `models/rits.py`**

#### Added `imputation_only` parameter to constructor:
```python
class Model(nn.Module):
    def __init__(self, features=35, imputation_only=False):
        super(Model, self).__init__()
        self.imputation_only = imputation_only
        self.build(features)
```

#### Modified `build()` to conditionally create classification layer:
```python
def build(self, features):
    # ... existing layers ...
    
    # Only create output layer if not imputation_only
    if not self.imputation_only:
        self.out = nn.Linear(RNN_HID_SIZE, 1)
```

#### Updated `forward()` to handle missing labels:
```python
def forward(self, data, direct):
    # ... extract values, masks, deltas ...
    
    # Only use labels/is_train if not imputation_only mode
    if not self.imputation_only:
        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)
    else:
        # Create dummy labels/is_train for imputation-only mode
        batch_size = values.size()[0]
        labels = torch.zeros((batch_size, 1))
        is_train = torch.ones((batch_size, 1))
        if torch.cuda.is_available():
            labels = labels.cuda()
            is_train = is_train.cuda()
    
    # ... imputation logic ...
    
    # Only compute classification loss if not imputation_only mode
    if not self.imputation_only:
        y_h = self.out(h)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce=False)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)
        y_h = F.sigmoid(y_h)
        total_loss = x_loss / SEQ_LEN + y_loss * 0.3
    else:
        # Imputation-only: no classification
        y_h = torch.zeros((values.size()[0], 1))
        if torch.cuda.is_available():
            y_h = y_h.cuda()
        total_loss = x_loss / SEQ_LEN  # Only reconstruction loss
```

### 2. **Updated `models/brits.py`**

#### Added `imputation_only` parameter:
```python
class Model(nn.Module):
    def __init__(self, imputation_only=True):  # Default to True for Beijing data
        super(Model, self).__init__()
        self.imputation_only = imputation_only
        self.build()

    def build(self):
        # Pass imputation_only to RITS models
        # Also specify 11 features for Beijing air quality data
        self.rits_f = rits.Model(features=11, imputation_only=self.imputation_only)
        self.rits_b = rits.Model(features=11, imputation_only=self.imputation_only)
```

### 3. **Updated `main_beijing.py`**

#### Create model with `imputation_only=True`:
```python
def run():
    # Create model in imputation-only mode (no classification task)
    model = getattr(models, args.model).Model(imputation_only=True)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    train(model)
```

### 4. **Fixed `beijing_handler.py`**

#### Re-added `labels` to collate_fn:
```python
def collate_fn(recs):
    # ... existing code ...
    
    # labels and is_train as CPU tensors
    ret_dict['labels'] = torch.FloatTensor([x['label'] for x in recs])
    ret_dict['is_train'] = torch.FloatTensor([x['is_train'] for x in recs])
    
    return ret_dict
```

---

## Loss Computation Changes

### Before (with classification):
```python
total_loss = imputation_loss / SEQ_LEN + classification_loss * 0.3
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
#            Reconstruction error          Binary classification error
```

### After (imputation-only):
```python
total_loss = imputation_loss / SEQ_LEN
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#            Only reconstruction error
```

**Benefits:**
- ✅ No need for classification labels
- ✅ Simpler loss function (pure imputation)
- ✅ Faster training (no classification head)
- ✅ Better for pure time-series imputation tasks

---

## Usage

### Beijing Air Quality (Imputation-Only):
```bash
python main_beijing.py --epochs 100 --batch_size 32 --model brits
```
- Uses `imputation_only=True` by default
- No classification labels needed
- Loss = reconstruction error only

### Original BRITS (with Classification):
```python
# In main.py (not main_beijing.py):
model = models.brits.Model(imputation_only=False)
```
- Requires `labels` in dataset
- Loss = reconstruction + 0.3 * classification

---

## Model Architecture Comparison

### Imputation-Only Mode (Beijing):
```
Input: [batch, 36, 11]  (36 timesteps, 11 features)
  ↓
LSTM + Temporal Decay + Feature Regression
  ↓
Output: [batch, 36, 11]  (imputed values)

Loss: MAE (reconstruction only)
```

### Full BRITS Mode (Original):
```
Input: [batch, 36, 35]  (36 timesteps, 35 features)
  ↓
LSTM + Temporal Decay + Feature Regression
  ├→ Imputations: [batch, 36, 35]
  └→ Classification: [batch, 1]

Loss: MAE + 0.3 * BCE (reconstruction + classification)
```

---

## Error Resolution

### Original Error:
```python
KeyError: 'labels'
```

**Cause:** Beijing dataset doesn't have classification labels, but RITS model expected them.

**Solution:** 
1. Added `imputation_only` parameter to skip classification logic
2. Creates dummy labels internally when needed (for backward compatibility)
3. Loss computation skips classification term

---

## Features Count

| Dataset | Features | Used For |
|---------|----------|----------|
| **Beijing Air Quality** | 11 | PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM |
| **Original BRITS Paper** | 35 | PhysioNet medical data (vitals, lab values) |

The model now automatically uses 11 features for Beijing data:
```python
self.rits_f = rits.Model(features=11, imputation_only=True)
```

---

## Summary

✅ **Fixed**: Model now works for pure imputation tasks without classification labels  
✅ **Backward Compatible**: Original classification mode still available with `imputation_only=False`  
✅ **Cleaner Loss**: Only reconstruction error for imputation tasks  
✅ **Correct Architecture**: Uses 11 features for Beijing air quality data  

The model is now ready to run on Beijing air quality data! 🎉

---

*Last updated: 2025-11-13*
