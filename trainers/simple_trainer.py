import os
from pathlib import Path
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
        task_name: str = None,
        run_id: str = None,
        save_results: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model','criterion','optimizer','test_metrics'])

        self.model = model
        self.criterion = criterion
        self._external_optimizer = optimizer
        self.test_metrics = nn.ModuleDict(test_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)
        self.task_name = task_name or "default_task"
        self.run_id = run_id or ""
        self.save_results = save_results

        # create results folder once
        self.results_dir = Path("results") / self.task_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def forward(self, **batch):
        return self.model(**batch)

    def _activation(self, logits):
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            return torch.sigmoid(logits)
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            return torch.softmax(logits, dim=-1)
        else:
            return logits

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        labels = batch["labels"].int()
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)

        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        probs = self._activation(logits)
        for metric in self.val_metrics.values():
            metric.update(probs, labels)

    def on_validation_epoch_end(self):
        # compute & log once per epoch
        out = []
        for name, metric in self.val_metrics.items():
            val = metric.compute()
            self.log(f"val_{name}", val, prog_bar=(name == "AUROC"))
            out.append(f"{name}={val:.4f}")
            metric.reset()

        if self.save_results:
            fn = self.results_dir / f"validation_results{self.run_id}.txt"
            with open(fn, "a") as f:
                f.write(f"Epoch {self.current_epoch}: " + ", ".join(out) + "\n")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # only update metrics, no compute/log here
        labels = batch["labels"].int()
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)
        loss = self.criterion(logits, labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        probs = self._activation(logits)
        for metric in self.test_metrics.values():
            metric.update(probs, labels)

    def on_test_epoch_end(self):
        # compute, log, print & optionally save once per run
        out = []
        print("\nFinal Test Results:\n")
        for name, metric in self.test_metrics.items():
            val = metric.compute()
            print(f"{name}: {val:.4f}")
            self.log(f"test_{name}", val, prog_bar=True)
            out.append(f"{name}={val:.4f}")
            metric.reset()

        if self.save_results:
            fn = self.results_dir / f"test_results{self.run_id}.txt"
            with open(fn, "a") as f:
                f.write(f"Run {self.run_id or self.current_epoch}: " + ", ".join(out) + "\n")

    def configure_optimizers(self):
        if self._external_optimizer is not None:
            return self._external_optimizer
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )