from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, data_dim, in_len, d_model, dim_feedforward, nhead, dropout, num_layers):
        super(TFC, self).__init__()

        self.embedding = nn.Linear(data_dim, d_model)

        encoder_layers_t = TransformerEncoderLayer(d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout)
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, num_layers)
        self.norm_t = nn.Linear(in_len * d_model, d_model)

        self.projector_t = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        encoder_layers_f = TransformerEncoderLayer(d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, num_layers)
        self.norm_f = nn.Linear(in_len * d_model, d_model)

        self.projector_f = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x_t = self.embedding(x_in_t)
        x_t = self.transformer_encoder_t(x_t)
        h_time = x_t.reshape(x_t.shape[0], -1)
        h_time = self.norm_t(h_time)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        x_f = self.embedding(x_in_f)
        x_f = self.transformer_encoder_f(x_f)
        h_freq = x_f.reshape(x_f.shape[0], -1)
        h_freq = self.norm_f(h_freq)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
