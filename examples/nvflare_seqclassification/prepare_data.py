#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preparation script for sequence classification.
This script reads multiple CSV files from a directory and combines the 'heavy' and 'light' feature columns
by concatenating them with a '|' separator. All processed data is combined into a single output CSV file.
"""

import os
import argparse
import pandas as pd
import glob


def prepare_data(input_dir, output_file, heavy_col='heavy', light_col='light', combined_col='combined'):
    """
    Read multiple CSV files from a directory, combine the 'heavy' and 'light' columns with a '|' separator,
    and save all processed data to a single output CSV file.
    
    Args:
        input_dir (str): Path to the directory containing input CSV files
        output_file (str): Path to save the output CSV file
        heavy_col (str): Name of the heavy column
        light_col (str): Name of the light column
        combined_col (str): Name for the new combined column
    """
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {input_dir}")
    
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    
    # Process each CSV file and combine the results
    all_data = []
    n_total = 0

    for i, input_file in enumerate(csv_files):
        print(f"Processing {input_file} ({i+1}/{len(csv_files)})...")
        
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check if required columns exist
        if heavy_col not in df.columns:
            print(f"Warning: Column '{heavy_col}' not found in {input_file}. Skipping this file.")
            continue
        if light_col not in df.columns:
            print(f"Warning: Column '{light_col}' not found in {input_file}. Skipping this file.")
            continue
        
        # Combine the columns with a '|' separator
        df[combined_col] = df[heavy_col].astype(str) + '|' + df[light_col].astype(str)
        
        print(f"Added {len(df)} rows...")
        n_total += len(df)
        
        # Add to the combined dataset
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No valid data found in any of the CSV files")
    
    # Combine all processed data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the result
    print(f"Saving processed data to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    print(f"Data preparation completed. Output saved {n_total} rows to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for sequence classification')
    parser.add_argument('--input_dir', '-i', required=True, help='Path to the directory containing input CSV files')
    parser.add_argument('--output_file', '-o', required=True, help='Path to save the output CSV file')
    
    args = parser.parse_args()
    
    prepare_data(
        args.input_dir,
        args.output_file
    )


if __name__ == "__main__":
    main()
