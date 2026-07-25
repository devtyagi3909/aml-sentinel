import pandas as pd
import numpy as np
import argparse
import os

def create_stratified_sample(input_path: str, output_path: str, target_size: int = 50000, seed: int = 42):
    print(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Total rows in raw dataset: {len(df):,}")
    
    # Isolate ALL rows matching the 6 engineered topologies
    if 'ground_truth_pattern' in df.columns:
        suspicious_df = df[df['ground_truth_pattern'] != 'none']
    elif 'ground_truth_flag' in df.columns:
        suspicious_df = df[df['ground_truth_flag'] == True]
    else:
        print("Warning: No ground truth flag found. Sampling randomly.")
        suspicious_df = pd.DataFrame()
        
    print(f"Found {len(suspicious_df):,} rows matching laundering topologies.")
    
    # Isolate normal rows
    normal_df = df[~df.index.isin(suspicious_df.index)]
    
    # Calculate how many normal rows we need
    needed_normal = target_size - len(suspicious_df)
    
    if needed_normal > 0:
        if needed_normal > len(normal_df):
            print(f"Warning: Not enough normal rows to reach target size {target_size}. Using all available.")
            sampled_normal = normal_df
        else:
            sampled_normal = normal_df.sample(n=needed_normal, random_state=seed)
        
        # Blend them together
        final_df = pd.concat([suspicious_df, sampled_normal]).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        print("Warning: Target size is smaller than suspicious rows. Returning only suspicious rows.")
        final_df = suspicious_df.sample(n=target_size, random_state=seed)
        
    print(f"Final stratified dataset size: {len(final_df):,}")
    
    # Print topology breakdown
    if 'ground_truth_pattern' in final_df.columns:
        breakdown = final_df[final_df['ground_truth_pattern'] != 'none']['ground_truth_pattern'].value_counts()
        print("\nTopology Breakdown in Demo Dataset:")
        print(breakdown)
        
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved demo dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a stratified 50k demo dataset")
    parser.add_argument('--input', type=str, default="sample_transactions.csv", help="Raw dataset path")
    parser.add_argument('--output', type=str, default="demo_transactions.csv", help="Output dataset path")
    parser.add_argument('--size', type=int, default=50000, help="Target size")
    args = parser.parse_args()
    
    create_stratified_sample(args.input, args.output, args.size)
