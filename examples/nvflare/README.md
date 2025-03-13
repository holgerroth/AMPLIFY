# NVFLARE Training Examples

This directory contains examples showing how to train MLP models using [NVFLARE](https://github.com/NVIDIA/NVFlare) for federated learning.

## Files

- `mlp.py` - Defines the MLP model architecture
- `dataloader.py` - Custom dataset loader for sequence data
- `train_mlp.py` - Standalone training script for single node training
- `train_mlp_fl.py` - Training script adapted for federated learning
- `run_fl.py` - Script to configure and launch federated training

## Usage

### Single Node Training

To train the MLP model on a single node:

1. Configure the paths in `train_mlp.py`:
   ```python
   model_dir = "/path/to/AMPLIFY/model"
   train_csv_file = "/path/to/training/data.csv"
   val_csv_file = "/path/to/validation/data.csv" 
   output_dir = "/path/to/output/directory"
   ```

2. Run the training script:
   ```bash
   python train_mlp.py
   ```

The script will:
- Load the AMPLIFY model for generating embeddings
- Create dataloaders for training and validation data
- Train the MLP model for the specified number of epochs
- Log metrics to TensorBoard
- Save the trained model

### Federated Learning

To run federated training across multiple clients:

1. Configure the federation parameters in `run_fl.py`:
   ```python
   n_clients = 2  # Number of clients
   num_rounds = 10  # Number of training rounds
   ```

2. Launch federated training:
   ```bash
   python run_fl.py
   ```

This will:
- Initialize the federated learning job configuration
- Set up the FedAvg controller for model aggregation
- Configure client training scripts
- Run the federated training simulation

The federated training uses the same MLP architecture but distributes training across multiple clients, with model aggregation happening after each round.

## Model Architecture

The MLP model:
- Takes protein sequence embeddings (dimension 640) as input
- Uses configurable hidden layers with ReLU activation and dropout
- Outputs binding affinity predictions

## Requirements

- PyTorch
- NVFLARE
- AMPLIFY
- TensorBoard
- pandas


## Visualizing Results

To visualize training metrics and model performance:

1. Launch TensorBoard:
   ```bash
   tensorboard --logdir /tmp/nvflare
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:6006
   ```

TensorBoard will display:
- Training and validation loss curves
- Model metrics over training rounds
- Performance comparisons across clients

The logs are saved in `/tmp/nvflare` by default. You can modify the log directory by changing the `output_dir` parameter in the training scripts.

