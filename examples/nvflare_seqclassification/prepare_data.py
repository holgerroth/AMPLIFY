#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preparation script for sequence classification.
This script reads a CSV file and combines the 'heavy' and 'light' feature columns
by concatenating them with a '|' separator.
"""

import os
import argparse
import pandas as pd


def prepare_data(input_file, output_file, heavy_col='heavy', light_col='light', combined_col='combined'):
    """
    Read a CSV file and combine the 'heavy' and 'light' columns with a '|' separator.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        heavy_col (str): Name of the heavy column
        light_col (str): Name of the light column
        combined_col (str): Name for the new combined column
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read the CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check if required columns exist
    if heavy_col not in df.columns:
        raise ValueError(f"Column '{heavy_col}' not found in the input file")
    if light_col not in df.columns:
        raise ValueError(f"Column '{light_col}' not found in the input file")
    
    # Combine the columns with a '|' separator
    print(f"Combining '{heavy_col}' and '{light_col}' columns...")
    df[combined_col] = df[heavy_col].astype(str) + '|' + df[light_col].astype(str)
    
    # Save the result
    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Data preparation completed. Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for sequence classification')
    parser.add_argument('--input', '-i', required=True, help='Path to the input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Path to save the output CSV file')
    parser.add_argument('--heavy-col', default='heavy', help='Name of the heavy column')
    parser.add_argument('--light-col', default='light', help='Name of the light column')
    parser.add_argument('--combined-col', default='combined', help='Name for the new combined column')
    
    args = parser.parse_args()
    
    prepare_data(
        args.input,
        args.output,
        args.heavy_col,
        args.light_col,
        args.combined_col
    )


if __name__ == "__main__":
    main()
