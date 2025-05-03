import torch
import torch.nn as nn
import math
from encoder.pos_encoding import PositionalEncoding

# Unimodal EHR Encoder: includes a CLS token.
class EHR_TSTEncoder(nn.Module):
    def __init__(self, input_dim, model_dim=64, max_time_len=4000,
                 nhead=2, ff_dim=128, nlayers=23, dropout=0.3):
        super(EHR_TSTEncoder, self).__init__()
        self.model_dim = model_dim
        self.in_projection = nn.Linear(input_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_encoding = PositionalEncoding(model_dim, dropout, max_len=max_time_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim,
                                                    activation='relu', batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, masks):
        # x: [batch, seq_len, input_dim]
        x = self.in_projection(x) * math.sqrt(self.model_dim)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, seq_len + 1, model_dim]
        x = self.pos_encoding(x)

        pad_col = torch.zeros(x.size(0), 1, dtype=masks.dtype, device=masks.device)
        masks = torch.cat([pad_col, masks], dim=1)

        x = self.transformer_encoder(x, src_key_padding_mask=masks)
        x = self.dropout(x)
        return x  # Full sequence: CLS + token embeddings