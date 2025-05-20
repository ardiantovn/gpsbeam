import torch
import torch.nn as nn
from typing import List

class BasePredModel(nn.Module):
    ARCH_NAME = 'base-pred-model'
    def __init__(self, 
                 feature_len: int, 
                 num_classes: int, 
                 mlp_hidden_layer_sizes: List[int] = [512,512]):
        super(BasePredModel, self).__init__()
        
        # MLP layers
        self.layers = nn.ModuleList()
        input_size = feature_len
        for size in mlp_hidden_layer_sizes:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size
        self.output = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output(x)
        return x