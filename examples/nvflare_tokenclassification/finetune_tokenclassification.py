import numpy as np

import torch
from torch import nn

from datasets import load_dataset
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification

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
        return self.classifier(h)
    
# Hyper-parameters
n_epochs = 2
batch_size = 8
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build Classifier on top of AMPLIFY 350M
model = amplify_classifier("chandar-lab/AMPLIFY_350M", 128, 3)
model = model.to(device)

# Load AMPLIFY tokenizer
tokenizer = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)

# Load the NetsurfP-2.0
dataset = load_dataset("biomap-research/ssp_q3")
dataset = dataset.rename_column("label", "labels")

# Set tokenizer
dataset.set_transform(lambda x: {"labels": x["labels"]} | tokenizer(x["seq"], padding=True, pad_to_multiple_of=8, return_tensors='pt'))

# Create the dataloaders
collate_fn = DataCollatorForTokenClassification(tokenizer, padding=True)
dataloader_train = torch.utils.data.DataLoader(dataset["train"], collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8)
dataloader_test = torch.utils.data.DataLoader(dataset["test"], collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, num_workers=8)

# Build the loss, optimizer, and scheduler
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=len(dataloader_train) * (n_epochs-1))

# Training loop
for epoch in range(n_epochs):
    train_loss, train_accuracy = [], []
    for i, batch in enumerate(dataloader_train):
        # Convert to correct dtype and move to GPU
        input_ids = batch["input_ids"].to(torch.long).to("cuda")
        attention_mask = batch["attention_mask"].to(torch.float32).to("cuda")
        labels = batch["labels"].to(torch.long).to("cuda")
        
        # Convert the attention mask to an additive mask
        attention_mask = torch.where(attention_mask==1, float(0.0), float("-inf"))
        
        # The AMPLIFY trunk is frozen_trunk during the first epoch
        output = model(input_ids, attention_mask, frozen_trunk=(epoch==0))
        
        # Compute the loss and accuracy
        loss = loss_fn(output.view(-1, 3), labels.view(-1))
        acc = torch.sum(torch.argmax(output, dim=-1) == labels) / (attention_mask != 0).sum()
        
        # Update the parameters
        loss.backward()
        optimizer.step()
        
        # Decay the learning rate only after the first epoch
        if epoch > 0:
            scheduler.step()
        optimizer.zero_grad()
        
        # Log the loss and accuracy
        train_loss.append(loss.item())
        train_accuracy.append(acc.item())
        print(f"\rEpoch: {epoch} Step {i:6d}/{len(dataloader_train)} loss: {np.mean(train_loss):.3f} acc: {np.mean(train_accuracy):.1%} lr: {scheduler.get_lr()}", end="")
        if i == 100:
            break
    
    # Evaluate
    with torch.no_grad():
        test_loss, test_accuracy = [], []
        for batch in dataloader_test:
            # Convert to correct dtype and move to GPU
            input_ids = batch["input_ids"].to(torch.long).to("cuda")
            attention_mask = batch["attention_mask"].to(torch.float32).to("cuda")
            labels = batch["labels"].to(torch.long).to("cuda")
        
            attention_mask = torch.where(attention_mask==1, float(0.0), float("-inf"))
            output = model(input_ids, attention_mask)
            loss = loss_fn(output.view(-1, 3), labels.view(-1))
            acc = torch.sum(torch.argmax(output, dim=-1) == labels) / (attention_mask != 0).sum()

            test_loss.append(loss.item())
            test_accuracy.append(acc.item())
            
        print(f"\n>>> Test loss: {np.mean(test_loss):.3f} acc: {np.mean(test_accuracy):.1%}")
        