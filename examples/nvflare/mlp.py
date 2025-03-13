import torch

# Define MLP model
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout_rate=0.1, output_dim=1):
        """
        Args:
            input_dim (int): Dimension of input features (embedding dimension)
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout rate for regularization
            output_dim (int): Dimension of output layer (default: 1 for regression)
        """
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
