import torch
import torch.nn as nn
from .pos_encoding import PositionalEncoding

class TSTModelECG_Patched(nn.Module):
    def __init__(self, input_dim, patch_size, model_dim=64, max_time_len=5000,
                 nhead=4, ff_dim=128, nlayers=3, dropout=0.3):
        super(TSTModelECG_Patched, self).__init__()
        assert max_time_len % patch_size == 0, "max_time_len must be divisible by patch_size"
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.max_num_patches = max_time_len // patch_size
        # Patch embedding: flatten patches and project to model_dim
        self.patch_embedding = nn.Linear(patch_size * input_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_encoding = PositionalEncoding(model_dim, dropout, max_len=self.max_num_patches + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim,
                                                    activation='relu', batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        # Pad if needed
        if seq_len % self.patch_size != 0:
            padding_size = self.patch_size - (seq_len % self.patch_size)
            x = torch.nn.functional.pad(x, (0, 0, 0, padding_size), mode='constant', value=0)
            seq_len = x.shape[1]
        num_patches = seq_len // self.patch_size
        # Reshape into patches
        x = x.view(batch_size, num_patches, -1)  # [batch, num_patches, patch_size*input_dim]
        x = self.patch_embedding(x)
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        return x  # Full sequence: CLS + token embeddings