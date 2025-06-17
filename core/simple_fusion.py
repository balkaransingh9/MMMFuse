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
        dropout: float = 0.2
    ):
        """
        encoders:        mapping modality name → encoder module
        embedding_dims:  mapping modality name → output dim of that encoder
        common_dim:      dimensionality of the shared space
        num_classes:     number of output targets
        dropout:         dropout rate after fusion
        """
        super().__init__()
        self.modalities = list(encoders.keys())
        self.encoders    = nn.ModuleDict(encoders)

        self.projections = nn.ModuleDict({
            m: nn.Linear(embedding_dims[m], common_dim)
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
            emb = self.encoders[m](inputs[m])                 # → (B, embedding_dims[m])
            emb = emb * present_mask[:, i].unsqueeze(-1)
            emb = self.projections[m](emb)                    # → (B, common_dim)
            projected.append(emb)

        x = torch.cat(projected, dim=1)                        # → (B, common_dim * M)
        x = self.fusion(x)                                     # → (B, common_dim)
        out = self.classifier(x)                               # → (B, num_classes)
        return out