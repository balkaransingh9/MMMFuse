import torch
import pytorch_lightning as pl
from torch.optim import Optimizer

class SimpleTrainer(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: Optimizer = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion', 'optimizer'])        
        self.model     = model
        self.criterion = criterion
        self._external_optimizer = optimizer

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)
        loss   = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)
        loss   = self.criterion(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Return the external optimizer if provided
        if self._external_optimizer is not None:
            return self._external_optimizer
        
        # Otherwise create a default one
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )