import torch
import pytorch_lightning as pl
from torch.optim import Optimizer

class SimpleTrainer(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: Optimizer = None,
        test_metrics = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5, 
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion', 'optimizer'])        
        self.model     = model
        self.criterion = criterion
        self._external_optimizer = optimizer
        self.test_metrics = test_metrics

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

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = self(**inputs)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)

        for name, metric in self.test_metrics.items():
            metric.update(probs, labels.int())
            self.log(f"test_{name}", metric.compute(), prog_bar=True)
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"logits": logits, "labels": labels}
    
    def on_test_epoch_end(self, outputs):
        print("\n Final Results: \n")
        for name, metric in self.test_metrics.items():
            score = metric.compute()
            print(f"{name}: {score:.4f}")
            metric.reset()

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