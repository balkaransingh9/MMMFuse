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
        # Save hyperparameters except modules/metrics themselves
        self.save_hyperparameters(ignore=['model','criterion','optimizer','test_metrics','val_metrics'])

        self.model = model
        self.criterion = criterion
        self._external_optimizer = optimizer

        # Wrap metrics in ModuleDict so they’re properly registered
        self.test_metrics = nn.ModuleDict(test_metrics or {})
        self.val_metrics  = nn.ModuleDict(val_metrics or {})

        self.task_name   = task_name or "default_task"
        self.run_id      = run_id or ""
        self.save_results = save_results

        # Prepare a directory for saving results
        self.results_dir = Path("results") / self.task_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # **Always** cast labels to long so metrics get integer targets
        self._cast_labels = lambda x: x.long()

    def forward(self, **batch):
        return self.model(**batch)

    def _activation(self, logits):
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            return torch.sigmoid(logits)
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            return torch.softmax(logits, dim=-1)
        return logits

    def _compute_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def _update_metrics(self, logits, labels, metrics_dict):
        probs = self._activation(logits)
        labels = self._cast_labels(labels)
        for metric in metrics_dict.values():
            metric.update(probs, labels)

    def _compute_and_log_metrics(self, metrics_dict, prefix: str):
        out = []
        for name, metric in metrics_dict.items():
            val = metric.compute()
            # show AUROC in progress bar if present
            self.log(f"{prefix}_{name}", val, prog_bar=(name == "AUROC"))
            out.append(f"{name}={val:.4f}")
            metric.reset()
        return out

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)

        loss = self._compute_loss(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)

        loss = self._compute_loss(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self._update_metrics(logits, labels, self.val_metrics)

    def on_validation_epoch_end(self):
        out = self._compute_and_log_metrics(self.val_metrics, prefix="val")
        if self.save_results:
            fn = self.results_dir / f"validation_results{self.run_id}.txt"
            with open(fn, "a") as f:
                f.write(f"Epoch {self.current_epoch}: " + ", ".join(out) + "\n")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)

        loss = self._compute_loss(logits, labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._update_metrics(logits, labels, self.test_metrics)

    def on_test_epoch_end(self):
        print("\nFinal Test Results:\n")
        out = self._compute_and_log_metrics(self.test_metrics, prefix="test")
        for line in out:
            print(line)
        if self.save_results:
            fn = self.results_dir / f"test_results_run{self.run_id}.txt"
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