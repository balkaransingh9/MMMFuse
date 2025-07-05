import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer

class SimpleTrainer(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: Optimizer = None,
        test_metrics: dict = None,
        val_metrics: dict = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion', 'optimizer', 'test_metrics'])

        self.model = model
        self.criterion = criterion
        self._external_optimizer = optimizer
        self.test_metrics = nn.ModuleDict(test_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)

    def forward(self, **batch):
        return self.model(**batch)

    def _activation(self, logits):
        """Applies correct activation based on loss function."""
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            return torch.sigmoid(logits)
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            return torch.softmax(logits, dim=-1)
        else:
            return logits  # fallback for regression, etc.

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)
        loss = self.criterion(logits, labels)

        probs = self._activation(logits)
        for name, metric in self.val_metrics.items():
            metric.update(probs, labels.int())
            self.log(f"val_{name}", metric.compute(), prog_bar=(name == "AUROC"))

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        for name, metric in self.val_metrics.items():
            metric.reset()


    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)
        loss = self.criterion(logits, labels)

        probs = self._activation(logits)
        for name, metric in self.test_metrics.items():
            metric.update(probs, labels.int())
            self.log(f"test_{name}", metric.compute(), prog_bar=True)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"logits": logits, "labels": labels}

    def on_test_epoch_end(self):
        print("\nFinal Test Results:\n")
        for name, metric in self.test_metrics.items():
            score = metric.compute()
            print(f"{name}: {score:.4f}")
            metric.reset()

    def configure_optimizers(self):
        if self._external_optimizer is not None:
            return self._external_optimizer
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )