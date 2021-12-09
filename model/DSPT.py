import torch
import torch.nn as nn
import torch.nn.functional as F
from .pos_encod import PositionalEncoding


class DSPT(nn.Module):
    def __init__(self):
        # add args dict
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(start_dim=2),
        )

        # # Encoder
        self.pe = PositionalEncoding(d_model=900, max_len=64)
        self.encoder = nn.ModuleList()
        self.num_trans_layers = 5
        for _ in range(self.num_trans_layers):
            self.encoder.append(nn.TransformerEncoderLayer(d_model=900, nhead=9, dim_feedforward=512, dropout=0,
                                layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None))

        # Decoder
        self.decoder = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.p1(x)
        x = self.pe(x)
        
        for i in range(self.num_trans_layers):
            x = self.encoder[i](x)
        x = self.decoder(x)
        return x
