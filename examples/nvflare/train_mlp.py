import os
import amplify
import torch
from torch import Tensor
from mlp import MLP
from dataloader import SequenceDataset
from torch.utils.tensorboard import SummaryWriter
import datetime

#model_dir = "/local/path/to/model"
model_dir = "/localhome/local-hroth/Data/AMPLIFY/AMPLIFY_120M"
train_csv_file = "/localhome/local-hroth/Data/AMPLIFY/FLAb/data/binding/Koenig2017_g6_Kd.csv"
val_csv_file = "/localhome/local-hroth/Data/AMPLIFY/FLAb/data/binding/Koenig2017_g6_Kd.csv"
output_dir = "/localhome/local-hroth/Data/AMPLIFY/nvflare_outputs"

BATCH_SIZE = 128
NUM_EPOCHS = 10

# Set up output directories
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
run_dir = os.path.join(output_dir, f'run_{current_time}')
os.makedirs(run_dir, exist_ok=True)

# Set up TensorBoard writer
writer = SummaryWriter(os.path.join(run_dir, 'logs'))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
config_path = os.path.join(model_dir, "config.yaml")
checkpoint_file = os.path.join(model_dir, "pytorch_model.pt")

model, tokenizer = amplify.AMPLIFY.load(checkpoint_file, config_path)

# Link the model to the inference API:
predictor = amplify.inference.Predictor(model, tokenizer, device=device)

# Create dataset and dataloader
train_sequence_dataset = SequenceDataset(train_csv_file, sequence_columns=["heavy", "light"], label_column="fitness")
val_sequence_dataset = SequenceDataset(val_csv_file, sequence_columns=["heavy", "light"], label_column="fitness")
                                   
train_dataloader = torch.utils.data.DataLoader(
    train_sequence_dataset,
    batch_size=BATCH_SIZE,  # Use constant defined at top of script
    shuffle=True,
    num_workers=4,  # Adjust based on your system
    pin_memory=True  # Helps with GPU transfer if using CUDA
)

val_dataloader = torch.utils.data.DataLoader(
    val_sequence_dataset,
    batch_size=BATCH_SIZE,  # Keep same batch size as training for consistency
    shuffle=False,   # No need to shuffle validation data
    num_workers=4,   # Keep same settings as training loader
    pin_memory=True
)

# Define the MLP model
model = MLP(input_dim=640)
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}]')
    epoch_loss = 0
    
    for batch_idx, (batch_sequences, batch_labels) in enumerate(train_dataloader):
        # Get embeddings for the batch
        batch_embeddings = []
        for heavy, light in zip(batch_sequences[0], batch_sequences[1]):
            # Assuming sequences is a list of [heavy, light] chains
            heavy_embedding = predictor.embed(heavy)
            light_embedding = predictor.embed(light)
            # Concatenate embeddings & take mean
            combined_embedding = torch.cat([Tensor(heavy_embedding), Tensor(light_embedding)], dim=0).mean(dim=0, keepdim=True)
            batch_embeddings.append(combined_embedding)
        
        # Stack embeddings into a batch and move to device
        batch_embeddings = torch.stack(batch_embeddings).to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        
        # Forward pass
        model.train()
        optimizer.zero_grad()
        outputs = model(batch_embeddings)
        loss = criterion(outputs.squeeze(), batch_labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track epoch loss
        epoch_loss += loss.item()
        
        # Log batch loss to TensorBoard
        global_step = epoch * len(train_dataloader) + batch_idx
        writer.add_scalar('Loss/batch', loss.item(), global_step)
        
        # Print batch loss
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
    
    # Calculate and log average epoch loss
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {avg_epoch_loss:.4f}')

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (batch_sequences, batch_labels) in enumerate(val_dataloader):
            # Get embeddings for the batch
            batch_embeddings = []
            for heavy, light in zip(batch_sequences[0], batch_sequences[1]):
                heavy_embedding = predictor.embed(heavy)
                light_embedding = predictor.embed(light)
                combined_embedding = torch.cat([Tensor(heavy_embedding), Tensor(light_embedding)], dim=0).mean(dim=0, keepdim=True)
                batch_embeddings.append(combined_embedding)
            
            batch_embeddings = torch.stack(batch_embeddings).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)   
            
            outputs = model(batch_embeddings)
            loss = criterion(outputs.squeeze(), batch_labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Loss: {avg_val_loss:.4f}')

# Close TensorBoard writer
writer.close()

# Save the trained model
model_save_path = os.path.join(run_dir, 'mlp_model.pt')
torch.save(model.state_dict(), model_save_path)
print(f"Training completed and model saved to {model_save_path}")
print(f"TensorBoard logs available in {os.path.join(run_dir, 'logs')}") 
