import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedFusionModel(nn.Module):
    """
    Hierarchical, uncertainty-gated multimodal fusion:
      1) Modality dropout for robustness
      2) Per-modality uncertainty gating
      3) Two-stage cross-attention fusion: physio+medicine, then with text
    """
    def __init__(
        self,
        encoders: dict,
        embedding_dims: dict,
        common_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.modalities = list(encoders.keys())
        self.num_modalities = len(self.modalities)
        self.encoders = nn.ModuleDict(encoders)

        # Linear projection + LayerNorm per modality
        self.projections = nn.ModuleDict({
            mod: nn.Linear(embedding_dims[mod], common_dim)
            for mod in self.modalities
        })
        self.norms = nn.ModuleDict({
            mod: nn.LayerNorm(common_dim)
            for mod in self.modalities
        })

        # Per-modality gating heads
        self.gating_mlps = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(common_dim, common_dim // 2),
                nn.ReLU(),
                nn.Linear(common_dim // 2, 1)
            ) for mod in self.modalities
        })

        # How often to randomly drop modalities at train time
        self.dropout_rate = dropout_rate

        # Cross-attention for physio+medicine
        self.cross_attn_pm = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Cross-attention for (physio+medicine) fused + text
        self.cross_attn_all = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, 1)
        )

    def forward(self, inputs: dict, present_mask: torch.Tensor):
        """
        inputs: dict of modality -> Tensor of shape (B, D_mod)
        present_mask: Tensor of shape (B, M) with 1=present, 0=missing
        """
        B, device = present_mask.size(0), present_mask.device

        # -- 1) Modality dropout to encourage robustness
        if self.training:
            drop_mask = torch.bernoulli(
                torch.full_like(present_mask, self.dropout_rate, device=device)
            )
            present_mask = present_mask * (1 - drop_mask)

        # -- 2) Encode, project, norm, zero-out missing
        embs = []
        for i, mod in enumerate(self.modalities):
            x = self.encoders[mod](inputs[mod])           # (B, D_mod)
            x = self.projections[mod](x)                  # (B, common_dim)
            x = self.norms[mod](x)                        # (B, common_dim)
            x = x * present_mask[:, i].unsqueeze(-1)      # mask missing
            embs.append(x)

        # -- 3) Uncertainty gating across modalities
        # compute raw scores
        scores = torch.cat([
            self.gating_mlps[mod](embs[i])
            for i, mod in enumerate(self.modalities)
        ], dim=1)           # (B, M)
        # mask out missing
        inf_mask = (present_mask == 0).float() * -1e9
        scores = scores + inf_mask
        weights = F.softmax(scores, dim=1)              # (B, M)
        # apply gating
        gated = [embs[i] * weights[:, i].unsqueeze(-1)
                 for i in range(self.num_modalities)]

        # -- 4) Hierarchical fusion: physio+medicine
        pm_idx = [self.modalities.index('physio'), self.modalities.index('medicine')]
        pm_seq = torch.stack([gated[i] for i in pm_idx], dim=1)  # (B, 2, common_dim)
        pm_fused, _ = self.cross_attn_pm(pm_seq, pm_seq, pm_seq)
        pm_repr = pm_fused.mean(dim=1)                            # (B, common_dim)

        # -- 5) Fuse with text
        text_idx = self.modalities.index('text')
        text_emb = gated[text_idx].unsqueeze(1)                   # (B, 1, common_dim)
        all_seq = torch.cat([pm_repr.unsqueeze(1), text_emb], dim=1)
        all_fused, _ = self.cross_attn_all(all_seq, all_seq, all_seq)
        fused_repr = all_fused[:, 0, :]                           # (B, common_dim)

        # -- 6) Classification
        out = self.classifier(fused_repr)                         # (B, 1)
        return out