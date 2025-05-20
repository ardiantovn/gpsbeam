import torch
import torch.nn as nn

from typing import List, Optional, Tuple
from loguru import logger

class CnnEdGruModel(nn.Module):
    ARCH_NAME = 'cnn-ed-gru-model'
    def __init__(self,
                cnn_channels: List[int] = [32, 64],
                feature_len: int = 2,
                num_classes: int = 64,
                rnn_num_layers: int = 1,
                rnn_hidden_size: int = 64,
                rnn_dropout: float = 0.0,
                cnn_dropout: float = 0.2,
                out_len: int = 3,
                mlp_layer_sizes: List[int] = None,
                enable_logging: bool = False):
        super(CnnEdGruModel, self).__init__()
        self.feature_len = feature_len        
        self.out_len = out_len
        self.rnn_hidden_size = rnn_hidden_size
        self.enable_logging = enable_logging

        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = feature_len
        for i, channels in enumerate(cnn_channels):
            self.conv_layers.append(
                nn.Conv1d(in_channels=in_channels, 
                        out_channels=channels, 
                        kernel_size=3, 
                        padding=1)
            )
            in_channels = channels
            
        # Only add batchnorm to last conv layer
        self.bn = nn.BatchNorm1d(cnn_channels[-1])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cnn_dropout)

        # Encoder GRU layer
        self.rnn_num_layers = rnn_num_layers
        self.encoder_rnn = nn.GRU(input_size=cnn_channels[-1],
                                hidden_size=self.rnn_hidden_size,
                                num_layers=rnn_num_layers,
                                dropout=rnn_dropout,
                                batch_first=True)
        
        # Decoder GRU layer
        self.decoder_rnn = nn.GRU(input_size=self.rnn_hidden_size,
                                hidden_size=self.rnn_hidden_size,
                                num_layers=rnn_num_layers,
                                dropout=rnn_dropout,
                                batch_first=True)
        
        # MLP layers
        input_size = self.rnn_hidden_size
        layers = []
        if mlp_layer_sizes is not None:
            # Build MLP with specified layer sizes
            for size in mlp_layer_sizes:
                layers.append(nn.Linear(input_size, size))
                layers.append(nn.ReLU())
                input_size = size
        # Final output layer
        layers.append(nn.Linear(input_size, num_classes))
        self.mlp_layers = nn.Sequential(*layers)
        

    def init_hidden(self, 
                    batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros((self.rnn_num_layers, 
                            batch_size, 
                            self.rnn_hidden_size), device=device)

    def encode(self, 
            x: torch.Tensor,
            hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder GRU forward pass
        encoder_output, encoder_hidden = self.encoder_rnn(x, hidden)
        return encoder_output, encoder_hidden

    def decode(self, 
            decoder_input: torch.Tensor,
            encoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize decoder hidden state
        decoder_hidden = encoder_hidden
        
        # Run decoder GRU on expanded input
        decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
        
        return decoder_output, decoder_hidden

    def classify(self,
                decoder_output: torch.Tensor) -> torch.Tensor:
        # Apply MLP layers
        return self.mlp_layers(decoder_output)
    
    def forward(self, 
                x: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        if self.enable_logging:
            logger.info(f"x shape: {x.shape}")

        # CNN forward pass
        # Input shape: [batch, seq_len, feature_len]
        # Reshape for CNN: [batch, feature_len, seq_len]
        x = x.transpose(1, 2)

        if self.enable_logging:
            logger.info(f"x shape after reshape: {x.shape}")
        
        # Apply conv layers without batchnorm except last layer
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = self.relu(conv(x))
            x = self.dropout(x)
        
        # Apply last conv layer with batchnorm
        x = self.relu(self.bn(self.conv_layers[-1](x)))
        x = self.dropout(x)
            
        if self.enable_logging:
            logger.info(f"x shape after conv: {x.shape}")
        
        # Reshape back: [batch, seq_len, feature_len]
        x = x.transpose(1, 2)
        
        if self.enable_logging:
            logger.info(f"x shape after reshape: {x.shape}")
        
        # Encode the input sequence
        encoder_output, encoder_hidden = self.encode(x, hidden)
        
        if self.enable_logging:
            logger.info(f"encoder_output shape: {encoder_output.shape}")
            logger.info(f"encoder_hidden shape: {encoder_hidden.shape}")
        
        # Get last encoder hidden state as initial decoder input
        decoder_input = encoder_hidden[-1:].transpose(0, 1)

        if self.enable_logging:
            logger.info(f"decoder_input shape: {decoder_input.shape}")

        # Expand decoder input to match output length
        # Shape: [batch, out_len, hidden_size] 
        expanded_input = decoder_input.repeat(1, self.out_len+1, 1)
        if self.enable_logging:
            logger.info(f"expanded_input shape: {expanded_input.shape}")
        
        # Decode using the encoder's hidden state
        decoder_output, decoder_hidden = self.decode(decoder_input=expanded_input, 
                                                    encoder_hidden=encoder_hidden)
        
        if self.enable_logging:
            logger.info(f"decoder_output shape: {decoder_output.shape}")
            logger.info(f"decoder_hidden shape: {decoder_hidden.shape}")
        
        # Apply classification
        output = self.classify(decoder_output)
        
        if self.enable_logging:
            logger.info(f"output shape: {output.shape}")
        
        return output, decoder_hidden    
