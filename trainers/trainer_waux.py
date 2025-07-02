import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer

class SimpleTrainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer = None,
        test_metrics: dict = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        aux_weight: float = 0.3,
        print_metrics: bool = True,
        aux_criterion: nn.Module = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion', 'optimizer', 'test_metrics'])

        self.model = model
        self.criterion = criterion
        self._external_optimizer = optimizer
        self.test_metrics = nn.ModuleDict(test_metrics)
        self.lr = lr
        self.weight_decay = weight_decay
        self.aux_weight = aux_weight
        self.aux_criterion = aux_criterion if aux_criterion is not None else copy.deepcopy(criterion)
        self.print_metrics = print_metrics

        self.val_metrics = nn.ModuleDict({
            name: metric.clone() for name, metric in self.test_metrics.items()
        })

        modalities = getattr(self.model, 'modalities', [])
        self.aux_val_metrics = nn.ModuleDict({
            mod: nn.ModuleDict({
                name: metric.clone() for name, metric in self.test_metrics.items()
            }) for mod in modalities
        })
        self.aux_test_metrics = nn.ModuleDict({
            mod: nn.ModuleDict({
                name: metric.clone() for name, metric in self.test_metrics.items()
            }) for mod in modalities
        })

    def forward(self, **batch):
        return self.model(**batch)

    def _activation(self, logits):
        """Choose correct activation based on output size and criterion."""
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            return torch.sigmoid(logits)
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            return torch.softmax(logits, dim=-1)
        else:
            return logits  # identity fallback

    def _compute_aux_loss(self, aux_logits, labels):
        return sum(self.aux_criterion(aux, labels) for aux in aux_logits.values()) / len(aux_logits)

    def _log_metrics(self, logits, labels, metrics_dict, prefix: str):
        probs = self._activation(logits)
        for name, metric in metrics_dict.items():
            metric.update(probs, labels.int())
            self.log(f'{prefix}_{name}', metric.compute(), on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self(**inputs)

        main_logits = outputs['logits']
        aux_logits = outputs.get('aux_logits', {})

        loss_main = self.criterion(main_logits, labels)
        self.log('train_loss_main', loss_main, on_step=True, on_epoch=True, prog_bar=True)

        if aux_logits:
            loss_aux = self._compute_aux_loss(aux_logits, labels)
            self.log('train_loss_aux', loss_aux, on_step=True, on_epoch=True, prog_bar=True)
            loss = loss_main + self.aux_weight * loss_aux
        else:
            loss = loss_main

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self(**inputs)

        main_logits = outputs['logits']
        aux_logits = outputs.get('aux_logits', {})

        loss_main = self.criterion(main_logits, labels)
        self.log('val_loss_main', loss_main, on_epoch=True, prog_bar=True)

        if aux_logits:
            loss_aux = self._compute_aux_loss(aux_logits, labels)
            self.log('val_loss_aux', loss_aux, on_epoch=True, prog_bar=True)
            loss = loss_main + self.aux_weight * loss_aux
        else:
            loss = loss_main

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        self._log_metrics(main_logits, labels, self.val_metrics, prefix='val')
        for mod, aux in aux_logits.items():
            self._log_metrics(aux, labels, self.aux_val_metrics[mod], prefix=f'val_{mod}')

    def on_validation_epoch_end(self):
        if not self.print_metrics:
            for metric in self.val_metrics.values():
                metric.reset()
            for mod_metrics in self.aux_val_metrics.values():
                for metric in mod_metrics.values():
                    metric.reset()
            return

        print("\nValidation Metrics:")
        for name, metric in self.val_metrics.items():
            print(f"{name}: {metric.compute():.4f}")
            metric.reset()

        print("\nAuxiliary Validation Metrics:")
        for mod, metrics in self.aux_val_metrics.items():
            print(f"-- {mod} --")
            for name, metric in metrics.items():
                print(f"{mod}_{name}: {metric.compute():.4f}")
                metric.reset()

    def test_step(self, batch, batch_idx):
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self(**inputs)

        main_logits = outputs['logits']
        aux_logits = outputs.get('aux_logits', {})

        loss_main = self.criterion(main_logits, labels)
        if aux_logits:
            loss_aux = self._compute_aux_loss(aux_logits, labels)
            loss = loss_main + self.aux_weight * loss_aux
        else:
            loss = loss_main

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self._log_metrics(main_logits, labels, self.test_metrics, prefix='test')

        for mod, aux in aux_logits.items():
            self._log_metrics(aux, labels, self.aux_test_metrics[mod], prefix=f'test_{mod}')

    def on_test_epoch_end(self):
        if not self.print_metrics:
            for metric in self.test_metrics.values():
                metric.reset()
            for mod_metrics in self.aux_test_metrics.values():
                for metric in mod_metrics.values():
                    metric.reset()
            return

        print("\nTest Metrics:")
        for name, metric in self.test_metrics.items():
            print(f"{name}: {metric.compute():.4f}")
            metric.reset()

        print("\nAuxiliary Test Metrics:")
        for mod, metrics in self.aux_test_metrics.items():
            print(f"-- {mod} --")
            for name, metric in metrics.items():
                print(f"{mod}_{name}: {metric.compute():.4f}")
                metric.reset()

    def configure_optimizers(self):
        if self._external_optimizer is not None:
            return self._external_optimizer
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )