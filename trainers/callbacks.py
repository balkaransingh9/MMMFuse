import pytorch_lightning as pl

class UnfreezeTextCallback(pl.Callback):
    def __init__(
        self,
        unfreeze_epoch: int = 3,
        gradual: bool = False,
        layers_per_epoch: int = 1,
        max_unfreeze: int = None,
        show_parms: bool = False
    ):
        """
        Args:
          unfreeze_epoch:  epoch at which to start unfreezing
          gradual:         if False, unfreeze *all* at unfreeze_epoch; 
                           if True, unfreeze in chunks
          layers_per_epoch: how many layers to open each epoch (once epoch>=unfreeze_epoch)
          max_unfreeze:    optional maximum total layers to ever unfreeze
          show_parms:      print a count of trainable vs total params
        """
        super().__init__()
        self.unfreeze_epoch    = unfreeze_epoch
        self.gradual           = gradual
        self.layers_per_epoch  = layers_per_epoch
        self.max_unfreeze      = max_unfreeze
        self.show_parms        = show_parms

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # only run on rank 0
        if not trainer.is_global_zero:
            return

        epoch    = trainer.current_epoch
        text_enc = pl_module.model.encoders['text']
        layers   = text_enc.encoder.encoder.layer
        total    = len(layers)

        # non-gradual: single-shot at unfreeze_epoch
        if not self.gradual:
            if epoch == self.unfreeze_epoch:
                for p in text_enc.parameters():
                    p.requires_grad = True
                print(f"→ [Epoch {epoch}] Unfroze ALL text parameters", flush=True)

        # gradual: in chunks
        else:
            if epoch >= self.unfreeze_epoch:
                steps = epoch - self.unfreeze_epoch + 1  
                target = steps * self.layers_per_epoch
                if self.max_unfreeze is not None:
                    target = min(target, self.max_unfreeze)
                target = min(target, total)

                if target > 0:
                    for layer in layers[-target:]:
                        for p in layer.parameters():
                            p.requires_grad = True

                print(
                    f"→ [Epoch {epoch}] Unfroze top {target} / {total} text layers",
                    flush=True
                )

        if self.show_parms:
            total_params     = sum(p.numel() for p in text_enc.parameters())
            trainable_params = sum(p.numel() for p in text_enc.parameters() if p.requires_grad)
            print(
                f"   Text‐encoder params: {trainable_params:,} / {total_params:,} trainable",
                flush=True
            )