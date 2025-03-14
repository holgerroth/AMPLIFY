import numpy as np
import os
import datetime

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


class amplify_classifier(nn.Module):
    def __init__(self, pretrained_model_name_or_path, hidden_size, num_labels):
        super().__init__()
        self.trunk = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.trunk.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, frozen_trunk=True, normalize_hidden_states=True, layer_idx=-1):
        with torch.no_grad() if frozen_trunk else torch.enable_grad():
            h = self.trunk(input_ids, attention_mask, output_hidden_states=True).hidden_states[layer_idx]
            
        if normalize_hidden_states:
            h = torch.nn.functional.normalize(h, p=2, dim=-1)
            
        # take mean of the hidden states for sequence classification
        h = h.mean(dim=1)
        
        # apply classifier
        return self.classifier(h)
    
# Hyper-parameters
n_epochs = 100
batch_size = 32
learning_rate = 5e-4

output_dir = "/tmp/nvflare/amplify_finetune_seqclassification"

# Start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up output directories
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
run_dir = os.path.join(output_dir, f'run_{current_time}')
os.makedirs(run_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(os.path.join(run_dir, 'logs'))

pretrained_model="chandar-lab/AMPLIFY_350M"

# Build Classifier on top of AMPLIFY
model = amplify_classifier(pretrained_model_name_or_path=pretrained_model, hidden_size=256, num_labels=1)  # one output label for regression task
model = model.to(device)

# Load AMPLIFY tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

# FLab bindings
train_csv_file = "/localhome/local-hroth/Data/AMPLIFY/FLAb/data/binding/Koenig2017_g6_Kd.csv"
test_csv_file = "/localhome/local-hroth/Data/AMPLIFY/FLAb/data/binding/Koenig2017_g6_Kd.csv"

data_files = {"train": train_csv_file, "test": test_csv_file}
dataset = load_dataset("csv", data_files=data_files)

# Set tokenizer
dataset.set_transform(lambda x: {"labels": x["fitness"]} | tokenizer(x["heavy"], padding=True, pad_to_multiple_of=8, return_tensors='pt'))

# Create the dataloaders
collate_fn = DataCollatorWithPadding(tokenizer, padding=True)
dataloader_train = torch.utils.data.DataLoader(dataset["train"], collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8)
dataloader_test = torch.utils.data.DataLoader(dataset["test"], collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, num_workers=8)

# Build the loss, optimizer, and scheduler
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=len(dataloader_train) * (n_epochs-1))

# Training loop
for epoch in range(n_epochs):
    train_loss = []
    for i, batch in enumerate(dataloader_train):
        model.train()
        # Convert to correct dtype and move to GPU
        input_ids = batch["input_ids"].to(torch.long).to(device)
        attention_mask = batch["attention_mask"].to(torch.float32).to(device)
        labels = batch["labels"].to(device)
        
        # Convert the attention mask to an additive mask
        attention_mask = torch.where(attention_mask==1, float(0.0), float("-inf"))
        
        # The AMPLIFY trunk is frozen_trunk during the first epoch
        output = model(input_ids, attention_mask, frozen_trunk=(epoch==0))
        
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
        print(f"\rEpoch: {epoch} Step {i:6d}/{len(dataloader_train)} loss: {current_loss:.3f} lr: {scheduler.get_lr()}", end="")
        # Log training loss to TensorBoard
        global_step = epoch * len(dataloader_train) + i
        writer.add_scalar('Loss/train', current_loss, global_step)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], global_step)
        if i == 100:
            break
    
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
model_save_path = os.path.join(run_dir, 'mlp_model.pt')
torch.save(model.state_dict(), model_save_path)
print(f"Training completed and model saved to {model_save_path}")
print(f"TensorBoard logs available in {os.path.join(run_dir, 'logs')}") 
