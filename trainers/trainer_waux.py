import copy
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
        print_metrics: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion', 'optimizer', 'test_metrics'])
        self.model             = model
        self.criterion         = criterion
        self._external_optimizer = optimizer
        self.test_metrics      = nn.ModuleDict(test_metrics)
        self.lr                = lr
        self.weight_decay      = weight_decay
        self.aux_weight        = aux_weight
        self.aux_criterion     = nn.BCEWithLogitsLoss()
        self.print_metrics     = print_metrics

        # clone metrics for validation
        self.val_metrics = nn.ModuleDict({
            name: metric.clone() for name, metric in self.test_metrics.items()
        })
        # aux metrics per modality
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

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self(**inputs)

        main_logits = outputs['logits']
        aux_logits  = outputs.get('aux_logits', {})

        loss_main = self.criterion(main_logits, labels)
        self.log('train_loss_main', loss_main, on_step=True, on_epoch=True, prog_bar=True)

        if aux_logits:
            loss_aux = sum(self.aux_criterion(aux, labels) for aux in aux_logits.values())
            loss_aux = loss_aux / len(aux_logits)
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
        aux_logits  = outputs.get('aux_logits', {})

        # loss
        loss_main = self.criterion(main_logits, labels)
        loss = loss_main
        if aux_logits:
            loss_aux = sum(self.aux_criterion(aux, labels) for aux in aux_logits.values())
            loss_aux = loss_aux / len(aux_logits)
            loss = loss_main + self.aux_weight * loss_aux
            self.log('val_loss_aux', loss_aux, on_epoch=True, prog_bar=True)
        self.log('val_loss_main', loss_main, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        # update metrics
        probs_main = torch.sigmoid(main_logits)
        for name, metric in self.val_metrics.items():
            metric.update(probs_main, labels.int())
            self.log(f'val_{name}', metric.compute(), on_epoch=True, prog_bar=True)

        for mod, aux in aux_logits.items():
            probs_aux = torch.sigmoid(aux)
            for name, metric in self.aux_val_metrics[mod].items():
                metric.update(probs_aux, labels.int())
                self.log(f'val_{mod}_{name}', metric.compute(), on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if not self.print_metrics:
            # reset metrics only
            for metric in self.val_metrics.values():
                metric.reset()
            for mod_metrics in self.aux_val_metrics.values():
                for metric in mod_metrics.values():
                    metric.reset()
            return

        # print main metrics
        print("\n Validation Metrics: ")
        for name, metric in self.val_metrics.items():
            score = metric.compute()
            print(f"{name}: {score:.4f}")
            metric.reset()
        # print aux metrics
        print("\n Auxiliary Validation Metrics: ")
        for mod, metrics in self.aux_val_metrics.items():
            print(f"-- {mod} --")
            for name, metric in metrics.items():
                score = metric.compute()
                print(f"{mod}_{name}: {score:.4f}")
                metric.reset()

    def test_step(self, batch, batch_idx):
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self(**inputs)

        main_logits = outputs['logits']
        aux_logits  = outputs.get('aux_logits', {})

        # loss
        loss_main = self.criterion(main_logits, labels)
        loss = loss_main
        if aux_logits:
            loss_aux = sum(self.aux_criterion(aux, labels) for aux in aux_logits.values())
            loss_aux = loss_aux / len(aux_logits)
            loss = loss_main + self.aux_weight * loss_aux
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # update metrics
        probs_main = torch.sigmoid(main_logits)
        for name, metric in self.test_metrics.items():
            metric.update(probs_main, labels.int())
            self.log(f'test_{name}', metric.compute(), on_epoch=True, prog_bar=True)

        for mod, aux in aux_logits.items():
            probs_aux = torch.sigmoid(aux)
            for name, metric in self.aux_test_metrics[mod].items():
                metric.update(probs_aux, labels.int())
                self.log(f'test_{mod}_{name}', metric.compute(), on_epoch=True, prog_bar=True)

        return {'logits': main_logits, 'labels': labels}

    def on_test_epoch_end(self):
        if not self.print_metrics:
            # reset test metrics only
            for metric in self.test_metrics.values():
                metric.reset()
            for mod_metrics in self.aux_test_metrics.values():
                for metric in mod_metrics.values():
                    metric.reset()
            return

        print("\n Final Main Test Metrics: \n")
        for name, metric in self.test_metrics.items():
            score = metric.compute()
            print(f"{name}: {score:.4f}")
            metric.reset()

        print("\n Final Auxiliary Test Metrics: \n")
        for mod, metrics in self.aux_test_metrics.items():
            print(f"-- {mod} --")
            for name, metric in metrics.items():
                score = metric.compute()
                print(f"{mod}_{name}: {score:.4f}")
                metric.reset()

    def configure_optimizers(self):
        if self._external_optimizer is not None:
            return self._external_optimizer
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )