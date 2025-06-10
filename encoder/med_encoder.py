import torch
import torch.nn as nn
from .pos_encoding import PositionalEncoding

class MedEncoder(nn.Module):
    def __init__(
        self, label_vocab_size=263, unit_vocab_size=8, category_vocab_size=17, 
        label_emb_dim=16, unit_emb_dim=16, category_emb_dim=8, hours_emb_dim=8, value_emb_dim=8,
        d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1, pad_idx=0, max_len=500,
        return_cls=True):
        super().__init__()

        self.label_emb = nn.Embedding(label_vocab_size, label_emb_dim, padding_idx=pad_idx)
        self.unit_emb = nn.Embedding(unit_vocab_size, unit_emb_dim, padding_idx=pad_idx)
        self.cat_emb = nn.Embedding(category_vocab_size, category_emb_dim, padding_idx=pad_idx)
        self.hours_proj = nn.Linear(1, hours_emb_dim)
        self.value_proj = nn.Linear(1, value_emb_dim)

        total_emb_dim = label_emb_dim + unit_emb_dim + category_emb_dim + hours_emb_dim + value_emb_dim

        self.input_proj = nn.Linear(total_emb_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Padding index for mask
        self.pad_idx = pad_idx
        self.return_cls = return_cls
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: dict):
        # x: dict with keys 'hours', 'value', 'label', 'unit', 'category'
        hours    = x['hours']    # (batch, seq_len)
        value    = x['value']    # (batch, seq_len)
        label    = x['label']    # (batch, seq_len)
        unit     = x['unit']     # (batch, seq_len)
        category = x['category'] # (batch, seq_len)

        # Padding mask based on label padding index
        pad_mask = (label == self.pad_idx)

        # Embeddings for each feature
        hours_emb = self.hours_proj(hours.unsqueeze(-1))    # (batch, seq_len, hours_emb_dim)
        value_emb = self.value_proj(value.unsqueeze(-1))    # (batch, seq_len, value_emb_dim)
        label_emb = self.label_emb(label)                   # (batch, seq_len, label_emb_dim)
        unit_emb  = self.unit_emb(unit)                     # (batch, seq_len, unit_emb_dim)
        cat_emb   = self.cat_emb(category)                  # (batch, seq_len, category_emb_dim)

        # Concatenate and project
        x_emb = torch.cat([label_emb, unit_emb, cat_emb, hours_emb, value_emb], dim=-1)
        x_proj = self.input_proj(x_emb)                     # (batch, seq_len, d_model)

        # Prepend CLS token
        batch_size = x_proj.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x_proj = torch.cat([cls_tokens, x_proj], dim=1)         # (batch, seq_len+1, d_model)

        # Extend padding mask for CLS (never masked)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=pad_mask.device)
        pad_mask = torch.cat([cls_mask, pad_mask], dim=1)       # (batch, seq_len+1)

        # Add positional encoding
        x_pos = self.pos_enc(x_proj)
        # Transformer encoding
        x = self.transformer(x_pos, src_key_padding_mask=pad_mask)
        x = self.dropout(x)

        if self.return_cls == True:
            return x[:, 0, :]
        else:
            return x