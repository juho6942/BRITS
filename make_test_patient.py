import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import os

def json_to_three_csvs(json_folder='json', output_folder='Data/data_patient'):
    """
    Convert JSON files to three separate CSVs:
    1. data.csv - All patient data (with eval values visible)
    2. fake_missing_data.csv - Same data but eval values replaced with NaN
    3. fake_missing_mask.csv - Binary mask showing which values were artificially masked
    
    Args:
        json_folder: Path to folder containing JSON files
        output_folder: Path to save CSV files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the json file (single file named 'json')
    json_file = os.path.join(json_folder, 'json')
    
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        return
    
    print(f"Reading: {json_file}")
    print("-" * 60)
    
    all_data_rows = []
    
    # Read line by line (each line is a separate patient JSON record)
    with open(json_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line)
                    patient_id = line_num  # Use line number as patient ID
                    
                    # Process BRITS format
                    if 'forward' in data:
                        forward_data = data['forward']
                        
                        for t, timestep in enumerate(forward_data):
                            row = {'patient_id': patient_id, 'timestep': t}
                            
                            # Extract values and eval_masks
                            if 'evals' in timestep:
                                evals = timestep['evals']
                                if isinstance(evals, (list, np.ndarray)):
                                    for i, val in enumerate(evals):
                                        row[f'feature_{i}'] = val
                            
                            if 'eval_masks' in timestep:
                                eval_masks = timestep['eval_masks']
                                if isinstance(eval_masks, (list, np.ndarray)):
                                    for i, mask_val in enumerate(eval_masks):
                                        row[f'eval_mask_{i}'] = mask_val
                            
                            all_data_rows.append(row)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    continue
    
    if not all_data_rows:
        print("Error: No data extracted from JSON file")
        return
    
    print(f"Processed {line_num} patients")
    print(f"Total records: {len(all_data_rows)}")
    
    # Convert to DataFrame
    df_all = pd.DataFrame(all_data_rows)
    
    # Organize columns - Sort numerically instead of alphabetically
    base_cols = ['patient_id', 'timestep']
    feature_cols = sorted(
        [col for col in df_all.columns if col.startswith('feature_')],
        key=lambda x: int(x.split('_')[1])  # Sort by number
    )
    eval_mask_cols = sorted(
        [col for col in df_all.columns if col.startswith('eval_mask_')],
        key=lambda x: int(x.split('_')[2])  # Sort by number
    )
    
    print(f"Number of features: {len(feature_cols)}")
    
    # === CSV 1: Complete data (with ground truth) ===
    df_data = df_all[base_cols + feature_cols].copy()
    data_file = os.path.join(output_folder, 'data.csv')
    df_data.to_csv(data_file, index=False)
    print(f"\n✓ CSV 1 saved: {data_file}")
    print(f"  Shape: {df_data.shape}")
    print(f"  Description: Complete data with ground truth values")
    
    # === CSV 2: Fake missing data (eval values replaced with NaN) ===
    df_fake_missing = df_data.copy()
    
    # Replace values with NaN where eval_mask is 1
    for i, feature_col in enumerate(feature_cols):
        eval_mask_col = f'eval_mask_{i}'
        if eval_mask_col in df_all.columns:
            # Set to NaN where mask is 1 (artificially masked)
            mask_values = df_all[eval_mask_col].fillna(0).astype(int)
            df_fake_missing.loc[mask_values == 1, feature_col] = np.nan
    
    fake_missing_file = os.path.join(output_folder, 'fake_missing_data.csv')
    df_fake_missing.to_csv(fake_missing_file, index=False)
    print(f"\n✓ CSV 2 saved: {fake_missing_file}")
    print(f"  Shape: {df_fake_missing.shape}")
    print(f"  Description: Data with artificially masked values set to NaN")
    print(f"  Missing values: {df_fake_missing[feature_cols].isna().sum().sum()}")
    
    # === CSV 3: Eval mask (binary mask for artificial missing values) ===
    df_eval_mask = df_all[base_cols + eval_mask_cols].copy()
    
    # Rename columns to match feature names
    rename_dict = {f'eval_mask_{i}': f'feature_{i}' for i in range(len(feature_cols))}
    df_eval_mask.rename(columns=rename_dict, inplace=True)
    
    # Fill NaN with 0 and convert to int
    for col in df_eval_mask.columns:
        if col.startswith('feature_'):
            df_eval_mask[col] = df_eval_mask[col].fillna(0).astype(int)
    
    mask_file = os.path.join(output_folder, 'fake_missing_mask.csv')
    df_eval_mask.to_csv(mask_file, index=False)
    print(f"\n✓ CSV 3 saved: {mask_file}")
    print(f"  Shape: {df_eval_mask.shape}")
    print(f"  Description: Binary mask (1=artificially masked, 0=observed)")
    
    # Count masked values per feature column
    masked_counts = df_eval_mask[[col for col in df_eval_mask.columns if col.startswith('feature_')]].sum()
    print(f"  Artificially masked values: {masked_counts.sum()}")
    
    # Summary
    print("\n" + "="*60)
    print("✓ Conversion complete!")
    print("="*60)
    print(f"\nOutput files in: {output_folder}/")
    print(f"  1. data.csv              - Complete data ({df_data.shape})")
    print(f"  2. fake_missing_data.csv - With NaN for masked values ({df_fake_missing.shape})")
    print(f"  3. fake_missing_mask.csv - Binary mask ({df_eval_mask.shape})")
    print(f"\nUnique patients: {df_data['patient_id'].nunique()}")
    print(f"Total timesteps: {len(df_data)}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Show sample
    print("\n" + "="*60)
    print("Sample from data.csv (first 5 rows):")
    print("="*60)
    print(df_data.head())
    
    print("\n" + "="*60)
    print("Sample from fake_missing_data.csv (first 5 rows):")
    print("="*60)
    print(df_fake_missing.head())
    
    print("\n" + "="*60)
    print("Sample from fake_missing_mask.csv (first 5 rows):")
    print("="*60)
    print(df_eval_mask.head())
    
    return df_data, df_fake_missing, df_eval_mask


def main():
    """Main function to convert JSON to three CSVs."""
    print("="*60)
    print("JSON to CSV Converter - Three File Output")
    print("="*60)
    print("\nCreating:")
    print("  1. data.csv - Ground truth data")
    print("  2. fake_missing_data.csv - Data with artificial missing values (NaN)")
    print("  3. fake_missing_mask.csv - Mask showing which values are artificial\n")
    
    json_to_three_csvs(
        json_folder='json', 
        output_folder='Data/data_patient'
    )

if __name__ == '__main__':
    main()

    main()