"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        d_model: int = 512,
        batch_first: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        # Position encoding calculation
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:x.size(1)]  # Add positional encoding to the input
        else:
            x = x + self.pe[:x.size(0)].unsqueeze(1)  # Add positional encoding to the input
        x = self.dropout(x)
        return x
