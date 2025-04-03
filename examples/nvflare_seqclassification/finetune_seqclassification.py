import numpy as np
import os
import datetime
import argparse

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from model import AmplifyClassifier, print_model_info

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune AMPLIFY model for sequence classification')
    # Data paths
    parser.add_argument('--train_csv', type=str, 
                      default="/localhome/local-hroth/Data/AMPLIFY/FLAb/data/binding/Koenig2017_g6_Kd_combined.csv",
                      help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str,
                      default="/localhome/local-hroth/Data/AMPLIFY/FLAb/data/binding/Koenig2017_g6_Kd_combined.csv",
                      help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str,
                      default="/tmp/nvflare/amplify_finetune_seqclassification",
                      help='Directory to save model and logs')
    # Pretrained model
    parser.add_argument('--pretrained_model', type=str,
                      default="chandar-lab/AMPLIFY_350M",
                      help='Name or path of the pretrained AMPLIFY model')
    # Hyper-parameters    
    parser.add_argument('--n_epochs', type=int,
                      default=30,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                      default=32,
                      help='Batch size for training')
    parser.add_argument('--trunk_lr', type=float,
                      default=1e-4,
                      help='Learning rate for the AMPLIFY trunk')
    parser.add_argument('--classifier_lr', type=float,
                      default=5e-4,
                      help='Learning rate for the classifier layers')
    # Model architecture
    parser.add_argument('--layer_sizes', type=str,
                      default="128,256,512,1024",
                      help='Comma-separated list of layer sizes for the classifier MLP')
    # Training options
    parser.add_argument('--frozen_trunk', action='store_true',
                      help='Whether to freeze the AMPLIFY trunk during training')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up output directories
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_dir = os.path.join(args.output_dir, f'run_{current_time}')
    os.makedirs(run_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(run_dir, 'logs'))

    # Parse layer sizes from string to list of integers
    layer_sizes = [int(size) for size in args.layer_sizes.split(',')]

    # Build Classifier on top of AMPLIFY
    model = AmplifyClassifier(pretrained_model_name_or_path=args.pretrained_model, layer_sizes=layer_sizes, num_labels=1)  # one output label for regression task
    model = model.to(device)
    
    # Print model architecture and configuration
    print_model_info(model, layer_sizes, args)

    # Load AMPLIFY tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, trust_remote_code=True)

    # Add data files
    data_files = {"train": args.train_csv, "test": args.test_csv}
    dataset = load_dataset("csv", data_files=data_files)

    # Set tokenizer
    dataset.set_transform(lambda x: {"labels": x["fitness"]} | tokenizer(x["combined"], padding=True, pad_to_multiple_of=8, return_tensors='pt'))

    # Create the dataloaders
    collate_fn = DataCollatorWithPadding(tokenizer, padding=True)
    dataloader_train = torch.utils.data.DataLoader(dataset["train"], collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=8)
    dataloader_test = torch.utils.data.DataLoader(dataset["test"], collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=True, num_workers=8)

    # Build the loss, optimizer, and scheduler
    loss_fn = torch.nn.MSELoss()
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': model.trunk.parameters(), 'lr': args.trunk_lr},
        {'params': model.classifier.parameters(), 'lr': args.classifier_lr}
    ]
    
    # Create single optimizer with parameter groups
    optimizer = torch.optim.AdamW(param_groups)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=len(dataloader_train) * (args.n_epochs-1))

    # Training loop
    for epoch in range(args.n_epochs):
        train_loss = []
        for i, batch in enumerate(dataloader_train):
            model.train()
            # Convert to correct dtype and move to GPU
            input_ids = batch["input_ids"].to(torch.long).to(device)
            attention_mask = batch["attention_mask"].to(torch.float32).to(device)
            labels = batch["labels"].to(device)
            
            # Convert the attention mask to an additive mask
            attention_mask = torch.where(attention_mask==1, float(0.0), float("-inf"))
            
            # Compute the model output
            output = model(input_ids, attention_mask, frozen_trunk=args.frozen_trunk)
            
            # Compute the loss and accuracy
            loss = loss_fn(output.squeeze(), labels)
            
            # Update the parameters
            loss.backward()
            optimizer.step()
            
            # Decay the learning rate only after the first epoch
            if epoch > 0:
                scheduler.step()
            optimizer.zero_grad()
            
            # Log the loss and accuracy
            train_loss.append(loss.item())
            current_loss = np.mean(train_loss)
            print(f"\rEpoch: {epoch} Step {i:6d}/{len(dataloader_train)} loss: {current_loss:.3f} trunk_lr: {scheduler.get_last_lr()[0]:.2e} classifier_lr: {scheduler.get_last_lr()[1]:.2e}", end="")
            # Log training loss to TensorBoard
            global_step = epoch * len(dataloader_train) + i
            writer.add_scalar('Loss/train', current_loss, global_step)
            writer.add_scalar('LR/trunk', scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('LR/classifier', scheduler.get_last_lr()[1], global_step)
            writer.add_scalar('Epoch', epoch, global_step)
        
        # Evaluate
        with torch.no_grad():
            test_loss = []
            for batch in dataloader_test:
                model.eval()
                # Convert to correct dtype and move to GPU
                input_ids = batch["input_ids"].to(torch.long).to(device)
                attention_mask = batch["attention_mask"].to(torch.float32).to(device)
                labels = batch["labels"].to(torch.long).to(device)
            
                attention_mask = torch.where(attention_mask==1, float(0.0), float("-inf"))
                output = model(input_ids, attention_mask)
                loss = loss_fn(output.squeeze(), labels)

                test_loss.append(loss.item())
                
            mean_test_loss = np.mean(test_loss)
            print(f"\n>>> Test loss: {mean_test_loss:.3f}")
            # Log test loss to TensorBoard
            writer.add_scalar('Loss/test', mean_test_loss, epoch)
            
    # Close the TensorBoard writer
    writer.close()
            
    # Save the trained model
    model_save_path = os.path.join(run_dir, 'fine_tuned_model.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"Training completed and model saved to {model_save_path}")
    print(f"TensorBoard logs available in {os.path.join(run_dir, 'logs')}")

if __name__ == "__main__":
    main() 
