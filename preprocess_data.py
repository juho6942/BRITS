import os
import pandas as pd
import numpy as np
import random

def preprocess_data(data_dir, output_file, mask_file, missing_mask_file):
    files = [f for f in os.listdir(data_dir) if f.endswith('.psv')]
    files.sort() # Ensure deterministic order
    files = files[:5000]
    # 1. Find Max Sequence Length
    print("Finding max sequence length...")
    max_seq_len = 0
    # Optimization: Just count lines instead of parsing CSV
    for idx, f in enumerate(files):
        if idx % 5000 == 0:
            print(f"Scanned {idx} files...")
        with open(os.path.join(data_dir, f), 'r') as fp:
            # Count lines. Subtract 1 for header.
            lines = sum(1 for _ in fp)
            seq_len = lines - 1
            
            if seq_len > 50:
                continue

            if seq_len > max_seq_len:
                max_seq_len = seq_len
    
    print(f"Max sequence length found: {max_seq_len}")
    
    # Features to track (excluding static/labels)
    # We'll infer features from the first file
    first_df = pd.read_csv(os.path.join(data_dir, files[0]), sep='|')
    # Exclude typical non-feature columns if they exist, though PhysioNet PSVs usually just have features + Time
    # We'll assume all columns except 'Time' are features.
    # Actually, looking at p000030.psv, columns are HR, O2Sat, etc.
    # We should probably keep all columns except maybe 'Time' if we are making a time-step index.
    feature_cols = [c for c in first_df.columns if c != 'Time']
    
    # Columns to exclude from evaluation mask
    exclude_mask_cols = {'PatientID', 'TimeStep', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel'}
    maskable_indices = [i for i, c in enumerate(feature_cols) if c not in exclude_mask_cols]
    
    # Write header
    cols = ['PatientID', 'TimeStep'] + feature_cols
    pd.DataFrame(columns=cols).to_csv(output_file, index=False)
    pd.DataFrame(columns=cols).to_csv(mask_file, index=False)
    pd.DataFrame(columns=cols).to_csv(missing_mask_file, index=False)
    
    print(f"Processing {len(files)} files...")
    
    chunk_data = []
    chunk_masks = []
    chunk_missing = []
    chunk_size = 1000
    
    for idx, f in enumerate(files):
        if idx % 1000 == 0:
            print(f"Processed {idx} files...")
            
        patient_id = f.split('.')[0]
        file_path = os.path.join(data_dir, f)
        df = pd.read_csv(file_path, sep='|')
        
        # Extract values
        values = df[feature_cols].values
        seq_len = len(values)
        
        if seq_len > 50:
            continue
        
        # Pad to max_seq_len
        if seq_len < max_seq_len:
            padding = np.full((max_seq_len - seq_len, len(feature_cols)), np.nan)
            padded_values = np.vstack([values, padding])
        else:
            padded_values = values[:max_seq_len]
            
        # Create Eval Mask
        # Only mask values that are observed AND not in the excluded columns
        valid_mask = ~np.isnan(values)
        
        # Create a boolean mask for columns that are allowed to be masked
        col_mask = np.zeros(values.shape[1], dtype=bool)
        col_mask[maskable_indices] = True
        
        # Apply column filter: must be observed AND in a maskable column
        candidate_mask = valid_mask & col_mask[None, :]
        
        observed_indices = np.where(candidate_mask)
        num_observed = len(observed_indices[0])
        num_to_mask = int(num_observed * 0.1)
        
        patient_eval_mask = np.zeros(padded_values.shape)
        if num_to_mask > 0:
            mask_indices = np.random.choice(num_observed, num_to_mask, replace=False)
            rows = observed_indices[0][mask_indices]
            col_indices = observed_indices[1][mask_indices]
            patient_eval_mask[rows, col_indices] = 1
            
        # Create Missing Mask (1 for missing/padding, 0 for observed)
        patient_missing_mask = np.isnan(padded_values).astype(int)

        padded_values = np.nan_to_num(padded_values)
        
        # Create DataFrame for this patient
        df_patient = pd.DataFrame(padded_values, columns=feature_cols)
        df_patient['PatientID'] = patient_id
        df_patient['TimeStep'] = range(max_seq_len)
        
        df_mask = pd.DataFrame(patient_eval_mask, columns=feature_cols)
        df_mask['PatientID'] = patient_id
        df_mask['TimeStep'] = range(max_seq_len)

        df_missing = pd.DataFrame(patient_missing_mask, columns=feature_cols)
        df_missing['PatientID'] = patient_id
        df_missing['TimeStep'] = range(max_seq_len)
        
        # Reorder
        df_patient = df_patient[cols]
        df_mask = df_mask[cols]
        df_missing = df_missing[cols]
        
        chunk_data.append(df_patient)
        chunk_masks.append(df_mask)
        chunk_missing.append(df_missing)
        
        # Write chunk
        if len(chunk_data) >= chunk_size:
            pd.concat(chunk_data).to_csv(output_file, mode='a', header=False, index=False)
            pd.concat(chunk_masks).to_csv(mask_file, mode='a', header=False, index=False)
            pd.concat(chunk_missing).to_csv(missing_mask_file, mode='a', header=False, index=False)
            chunk_data = []
            chunk_masks = []
            chunk_missing = []

    # Write remaining
    if chunk_data:
        pd.concat(chunk_data).to_csv(output_file, mode='a', header=False, index=False)
        pd.concat(chunk_masks).to_csv(mask_file, mode='a', header=False, index=False)
        pd.concat(chunk_missing).to_csv(missing_mask_file, mode='a', header=False, index=False)
    
    print("Done.")

if __name__ == "__main__":
    data_dir = r"c:\Schoolwork\BIOMED\BRITS\Data\training_setA\training_setA"
    output_file = r"c:\Schoolwork\BIOMED\BRITS\Data\combined_patient_data.csv"
    mask_file = r"c:\Schoolwork\BIOMED\BRITS\Data\combined_eval_mask.csv"
    missing_mask_file = r"c:\Schoolwork\BIOMED\BRITS\Data\combined_missing_mask.csv"
    preprocess_data(data_dir, output_file, mask_file, missing_mask_file)
