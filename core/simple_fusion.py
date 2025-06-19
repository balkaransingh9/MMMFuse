import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFusion(nn.Module):
    def __init__(
        self,
        encoders: dict[str, nn.Module],
        embedding_dims: dict[str, int],
        common_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.2,
        use_layernorm: bool = False,
    ):
        """
        encoders:        mapping modality name → encoder module
        embedding_dims:  mapping modality name → output dim of that encoder
        common_dim:      dimensionality of the shared space
        num_classes:     number of output targets
        dropout:         dropout rate after fusion
        use_layernorm:   whether to apply LayerNorm to each projected modality
        """
        super().__init__()
        self.modalities    = list(encoders.keys())
        self.encoders      = nn.ModuleDict(encoders)
        self.use_layernorm = use_layernorm

        self.projections = nn.ModuleDict({
            m: nn.Linear(embedding_dims[m], common_dim)
            for m in self.modalities
        })

        if self.use_layernorm:
            self.layer_norms = nn.ModuleDict({
                m: nn.LayerNorm(common_dim)
                for m in self.modalities
            })

        fused_dim = common_dim * len(self.modalities)
        self.fusion     = nn.Linear(fused_dim, common_dim)
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(common_dim, num_classes)
        )

    def forward(self, inputs: dict[str, torch.Tensor], present_mask: torch.Tensor):
        """
        inputs:       mapping modality name → tensor of shape (B, embedding_dims[m])
        present_mask: (B, M) binary mask indicating which modalities are present
                      (M == number of modalities, in same order as self.modalities)
        """
        projected = []
        for i, m in enumerate(self.modalities):
            emb = self.encoders[m](inputs[m])              
            emb = emb * present_mask[:, i].unsqueeze(-1)  # zero‐out missing
            emb = self.projections[m](emb)                

            if self.use_layernorm:
                emb = self.layer_norms[m](emb)            
            projected.append(emb)

        # fuse
        x = torch.cat(projected, dim=1)
        x = self.fusion(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out
