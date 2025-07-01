import pytorch_lightning as pl

class UnfreezeTextCallback(pl.Callback):
    def __init__(self, unfreeze_epoch: int = 3, gradual: bool = False, 
                 show_parms: bool = False):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.gradual        = gradual
        self.show_parms     = show_parms

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return

        epoch    = trainer.current_epoch
        text_enc = pl_module.model.encoders['text']
        layers   = text_enc.encoder.encoder.layer

        if not self.gradual:
            if epoch == self.unfreeze_epoch:
                for p in text_enc.parameters():
                    p.requires_grad = True
                print(f"→ [Epoch {epoch}] Unfroze ALL text parameters", flush=True)
        else:
            idx = epoch - self.unfreeze_epoch
            if 0 <= idx < len(layers):
                layer = layers[-1 - idx]
                for p in layer.parameters():
                    p.requires_grad = True
                print(f"→ [Epoch {epoch}] Unfroze text layer {len(layers)-idx-1}", flush=True)

        if self.show_parms:
            total_params = sum(p.numel() for p in text_enc.parameters())
            trainable_params = sum(p.numel() for p in text_enc.parameters() if p.requires_grad)
            print(f"   Text‐encoder params: {trainable_params:,} / {total_params:,} trainable",
                flush=True)