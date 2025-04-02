import torch
from torch import nn
from transformers import AutoModel

class AmplifyClassifier(nn.Module):
    def __init__(self, pretrained_model_name_or_path, layer_sizes, num_labels, dropout_rate=0.1, num_groups=8):
        super().__init__()
        self.trunk = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        
        # Create classifier layers dynamically based on layer_sizes
        layers = []
        prev_size = self.trunk.config.hidden_size
        
        for size in layer_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.GroupNorm(num_groups, size),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = size
            
        # Add final layer to num_labels
        layers.append(nn.Linear(prev_size, num_labels))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, frozen_trunk=True, normalize_hidden_states=True, layer_idx=-1):
        with torch.no_grad() if frozen_trunk else torch.enable_grad():
            h = self.trunk(input_ids, attention_mask, output_hidden_states=True).hidden_states[layer_idx]
            
        if normalize_hidden_states:
            h = torch.nn.functional.normalize(h, p=2, dim=-1)
            
        # take mean of the hidden states for sequence classification
        h = h.mean(dim=1)
        
        # apply classifier
        return self.classifier(h) 