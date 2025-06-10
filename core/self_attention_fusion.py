import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalTransformerFusion(nn.Module):
    def __init__(
        self,
        encoders: dict,
        embedding_dims: dict,
        common_dim: int = 256,
        fused_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 25
    ):
        """
        Args:
            encoders:      {'text': TextEncoder, 'ecg': ECGEncoder, ...}
            embedding_dims:{'text':768, 'ecg':128, 'physio':64, ...}
            common_dim:    all embeddings are projected to this size
            fused_dim:     final output size
            num_heads:     attention heads in transformer
            num_layers:    number of transformer‐encoder layers
        """
        super().__init__()
        self.modalities = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)

        self.projections = nn.ModuleDict({
            m: nn.Linear(embedding_dims[m], common_dim)
            for m in self.modalities
        })

        self.cls_token = nn.Parameter(torch.empty(1, 1, common_dim))
        self.pos_embed = nn.Parameter(torch.empty(1, len(self.modalities) + 1, common_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=common_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(common_dim)
        self.fusion_head = nn.Linear(common_dim, fused_dim)

        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, inputs: dict, present_mask: torch.Tensor):
        """
        inputs:      {'text':Tensor[B,txt_dim], 'ecg':Tensor[B,T_ecg,ecg_dim], 'physio':Tensor[B,T_phy,phy_dim]}
        present_mask:FloatTensor[B, M] (1.0=present, 0.0=missing), in the same order as self.modalities
        """
        batch_size = present_mask.size(0)
        device     = present_mask.device
        reps = []

        for i, m in enumerate(self.modalities):
            x   = inputs[m]
            emb = self.encoders[m](x)  # might be [B, D] or [B, T, D]
            emb = self.projections[m](emb)  # → [B, common_dim]

            mask = present_mask[:, i].unsqueeze(-1)  # → [B,1]
            emb  = emb * mask
            reps.append(emb)

        # stack into tokens, prepend CLS, add pos, mask
        tokens = torch.stack(reps, dim=1)                     # [B, M, common_dim]
        cls    = self.cls_token.expand(batch_size, -1, -1)    # [B,1,common_dim]
        x      = torch.cat([cls, tokens], dim=1)              # [B, M+1, common_dim]
        x     += self.pos_embed

        # masking: True=ignore; CLS always present
        pad_mask = torch.zeros(batch_size, len(self.modalities)+1,
                               device=device, dtype=torch.bool)
        pad_mask[:,1:] = present_mask == 0

        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # pool from CLS
        cls_out = self.norm(x[:,0])               # [B,common_dim]
        cls_out = self.fusion_head(cls_out)
        cls_out = F.relu(cls_out)
        cls_out = self.classifier(cls_out)
        return cls_out