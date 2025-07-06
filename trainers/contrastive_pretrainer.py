import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import itertools

class ContrastivePretrainer(pl.LightningModule):
    def __init__(
        self,
        encoders: dict[str, nn.Module],
        embedding_dims: dict[str, int],
        projection_dim: int = 128,
        temp: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoders','optimizer'])
        self.modalities = list(encoders.keys())
        self.encoders   = nn.ModuleDict(encoders)
        self._external_optimizer = optimizer

        # one MLP head per modality
        self.projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(embedding_dims[m], projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )
            for m in self.modalities
        })
        self.temp = temp

    def info_nce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                       # (2B, D)
        sims = torch.matmul(z, z.T) / self.temp              # (2B,2B)
        mask = (~torch.eye(2*B, device=z.device)).float()
        exp_sims = torch.exp(sims) * mask
        pos = torch.exp((z1 * z2).sum(-1) / self.temp)       # (B,)
        pos = torch.cat([pos, pos], dim=0)                  # (2B,)
        return -torch.log(pos / exp_sims.sum(dim=-1)).mean()

    def training_step(self, batch, batch_idx):
        # assume batch includes keys for each modality plus:
        #   batch['present_mask'] shape (B, M)
        pres = batch['present_mask']   # Tensor of 0/1
        inputs = batch['inputs']
        feats = {}

        # 1) encode+project all modalities
        for i, m in enumerate(self.modalities):
            h = self.encoders[m](inputs[m])               # (B, E_m)
            p = self.projections[m](h)                   # (B, P)
            feats[m] = F.normalize(p, dim=-1)            # normalize now

        # 2) compute loss only over samples where both modalities exist
        losses = []
        for (i1, m1), (i2, m2) in itertools.combinations(enumerate(self.modalities), 2):
            valid = (pres[:, i1] & pres[:, i2]).bool()  # (B,)
            if valid.sum() < 2:
                continue  # not enough samples to form a batch
            z1 = feats[m1][valid]
            z2 = feats[m2][valid]
            losses.append(self.info_nce(z1, z2))

        if not losses:
            # no valid pairs this batch? skip grad
            loss = torch.tensor(0., device=self.device, requires_grad=True)
        else:
            loss = torch.stack(losses).mean()

        self.log("ctr_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self._external_optimizer is not None:
            return self._external_optimizer
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
