import torch
import torch.nn as nn
from typing import List, Optional

class BaseTrackModel(nn.Module):
    ARCH_NAME = 'base-track-model'
    def __init__(self, 
                 feature_len: int,
                 num_classes: int,
                 rnn_num_layers: int,
                 rnn_hidden_size: int,
                 rnn_dropout: float,
                 out_len: int):
        super(BaseTrackModel, self).__init__()
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=feature_len,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Linear(rnn_hidden_size, num_classes)
        
        # Save parameters
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.out_len = out_len

    def init_hidden(self, 
                    batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros((self.rnn_num_layers, 
                            batch_size, 
                            self.rnn_hidden_size), device=device)
        
    def forward(self, 
                x: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch_size, seq_len, feature_len)
        batch_size = x.size(0)
        # print(f" x shape: {x.shape}")

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # Pass through GRU
        out, hidden = self.gru(x, hidden)
        # out shape: (batch_size, seq_len, rnn_hidden_size)
        
        # print(f"BEFORE out shape: {out.shape}")

        # Take the last 'out_len' steps from the GRU output
        # out = out[:, -self.out_len:, :]
        # out shape: (batch_size, out_len, rnn_hidden_size)
        # print(f"AFTER out shape: {out.shape}")

        # Apply output layer to each time step
        predictions = self.output(out)
        
        return predictions, hidden