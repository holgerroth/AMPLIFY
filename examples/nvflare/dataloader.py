import torch
import pandas as pd

# Create a custom Dataset class
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str, sequence_columns=["heavy", "light"], label_column=None, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing the data
            sequence_columns (list): List of column names containing sequences
            label_column (str, optional): Name of the column containing labels
            transform (callable, optional): Optional transform to be applied on sequences
        """
        self.csv_file = csv_file
        self.sequence_columns = sequence_columns
        self.label_column = label_column
        self.transform = transform

        # Read data from CSV file
        self.data = pd.read_csv(csv_file)        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get sequences from all specified columns
        sequences = [self.data.iloc[idx][col] for col in self.sequence_columns]
        
        if self.transform:
            sequences = [self.transform(seq) for seq in sequences]

        # If label column exists, include it in the return
        if self.label_column is not None:
            label = self.data.iloc[idx][self.label_column]
            return sequences, label
            
        return sequences
