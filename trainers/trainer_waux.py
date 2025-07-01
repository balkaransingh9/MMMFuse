import torch
import pytorch_lightning as pl
from torch.optim import Optimizer
import torch.nn as nn

class TrainerWAux(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: Optimizer = None,
        test_metrics: dict = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        aux_weight: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion', 'optimizer', 'test_metrics'])
        self.model         = model
        self.criterion     = criterion
        self._external_optimizer = optimizer
        self.test_metrics  = nn.ModuleDict(test_metrics)
        # auxiliary heads criterion and weight
        self.aux_weight    = aux_weight
        self.aux_criterion = nn.BCEWithLogitsLoss()

    def forward(self, **batch):
        # model should return a dict: {'logits': main_logits, 'aux_logits': {'physio':..., 'text':..., ...}}
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self(**inputs)

        # unpack main and auxiliary predictions
        main_logits = outputs['logits']
        aux_logits  = outputs.get('aux_logits', {})

        # primary loss
        loss_main = self.criterion(main_logits, labels)
        self.log('train_loss_main', loss_main, on_step=True, on_epoch=True, prog_bar=True)

        # auxiliary loss (averaged)
        if aux_logits:
            loss_aux = 0.0
            for name, aux in aux_logits.items():
                loss_aux += self.aux_criterion(aux, labels)
                self.log(f'train_loss_aux_{name}', loss_aux, on_step=True, on_epoch=True)
            loss_aux = loss_aux / len(aux_logits)
            loss = loss_main + self.aux_weight * loss_aux
            self.log('train_loss_aux', loss_aux, on_step=True, on_epoch=True, prog_bar=True)
        else:
            loss = loss_main

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self(**inputs)

        main_logits = outputs['logits']
        aux_logits  = outputs.get('aux_logits', {})

        loss_main = self.criterion(main_logits, labels)
        self.log('val_loss_main', loss_main, on_epoch=True, prog_bar=True)

        if aux_logits:
            loss_aux = 0.0
            for name, aux in aux_logits.items():
                loss_aux += self.aux_criterion(aux, labels)
            loss_aux = loss_aux / len(aux_logits)
            self.log('val_loss_aux', loss_aux, on_epoch=True, prog_bar=True)
            loss = loss_main + self.aux_weight * loss_aux
        else:
            loss = loss_main

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self(**inputs)

        main_logits = outputs['logits']
        aux_logits  = outputs.get('aux_logits', {})

        loss_main = self.criterion(main_logits, labels)

        if aux_logits:
            loss_aux = 0.0
            for aux in aux_logits.values():
                loss_aux += self.aux_criterion(aux, labels)
            loss_aux = loss_aux / len(aux_logits)
            loss = loss_main + self.aux_weight * loss_aux
        else:
            loss = loss_main

        probs = torch.sigmoid(main_logits)
        for name, metric in self.test_metrics.items():
            metric.update(probs, labels.int())
            self.log(f'test_{name}', metric.compute(), prog_bar=True)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'logits': main_logits, 'labels': labels}

    def on_test_epoch_end(self):
        print("\n Final Results: \n")
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
