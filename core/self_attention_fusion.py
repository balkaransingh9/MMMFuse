import torch
import torch.nn as nn

class SelfAttentionFusion(nn.Module):
    def __init__(
        self,
        encoders: dict,         # e.g. {'physio': PhysioEncoder(), 'text': TextEncoder(), ...}
        embedding_dims: dict,   # matching dict of input dims, e.g. {'physio':128,'text':768,...}
        common_dim: int = 256,  # hidden size in fusion space
        num_classes: int = 1,   # output dim
        num_attention_heads: int = 4,
        num_transformer_layers: int = 1,
        dropout: float = 0.2,
        dropout_classifier: float = 0.2
    ):
        super().__init__()

        self.modalities = list(encoders.keys())
        self.num_modalities = len(self.modalities)
        self.encoders = nn.ModuleDict(encoders)

        self.projections = nn.ModuleDict({
            mod: nn.Linear(embedding_dims[mod], common_dim)
            for mod in self.modalities
        })
        self.norms = nn.ModuleDict({
            mod: nn.LayerNorm(common_dim)
            for mod in self.modalities
        })

        self.modality_embedding = nn.Embedding(self.num_modalities, common_dim)
        self.pos_embedding = nn.Embedding(self.num_modalities + 1, common_dim)
        self.fusion_token = nn.Parameter(torch.randn(1, 1, common_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=common_dim,
            nhead=num_attention_heads,
            dim_feedforward=common_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True)

        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers)

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(common_dim, num_classes)
        )

    def forward(self, inputs: dict, present_mask: dict[str, torch.Tensor]):
        """
        inputs: dict of modality → Tensor of shape (B, D_mod)
        present_mask: dict of modality → Tensor of shape (B,) with 1=present, 0=missing
        """
        # grab batch size and device from any one modality
        first_mod = self.modalities[0]
        B = present_mask[first_mod].size(0)
        device = present_mask[first_mod].device

        # encode each modality, zeroing out missing examples
        embedded = []
        for mod in self.modalities:
            emb = self.encoders[mod](inputs[mod])             # (B, E_mod)
            mask1d = present_mask[mod].unsqueeze(-1)          # (B, 1)
            emb = emb * mask1d                                # zero‐out missing
            emb = self.norms[mod](self.projections[mod](emb)) # (B, common_dim)
            embedded.append(emb)

        # stack modalities → (B, M, common_dim)
        modal_embs = torch.stack(embedded, dim=1)

        # add modality‐type embeddings
        type_idx = torch.arange(self.num_modalities, device=device) \
                        .unsqueeze(0).expand(B, -1)
        modal_embs = modal_embs + self.modality_embedding(type_idx)

        # prepend fusion token
        fusion_tok = self.fusion_token.expand(B, -1, -1)       # (B, 1, common_dim)
        seq_embs = torch.cat([fusion_tok, modal_embs], dim=1)  # (B, M+1, common_dim)

        # add positional embeddings
        pos_idx = torch.arange(self.num_modalities + 1, device=device) \
                       .unsqueeze(0).expand(B, -1)
        seq_embs = seq_embs + self.pos_embedding(pos_idx)

        # build key_padding_mask for transformer: True = mask (i.e. missing)
        fusion_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        modal_mask = torch.stack([
            present_mask[mod] == 0
            for mod in self.modalities
        ], dim=1)  # (B, M)
        key_padding_mask = torch.cat([fusion_mask, modal_mask], dim=1)  # (B, M+1)

        # fuse with transformer
        fused = self.fusion_transformer(
            src=seq_embs,
            src_key_padding_mask=key_padding_mask
        )

        # classify on fusion token
        fused_repr = fused[:, 0, :]       # (B, common_dim)
        out = self.classifier(fused_repr) # (B, num_classes)
        return out