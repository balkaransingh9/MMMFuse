import pytorch_lightning as pl

class UnfreezeTextCallback(pl.Callback):
    def __init__(self, unfreeze_epoch: int = 3, gradual: bool = False):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.gradual        = gradual

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

        total_params = sum(p.numel() for p in text_enc.parameters())
        trainable_params = sum(p.numel() for p in text_enc.parameters() if p.requires_grad)
        print(f"   Text‐encoder params: {trainable_params:,} / {total_params:,} trainable",
              flush=True)

        # 3) (Optional) Show a few parameter names and their requires_grad flags
        #    so you can spot‐check that e.g. layer 11 is now trainable.
        printed = 0
        for name, p in text_enc.named_parameters():
            if "encoder.layer." in name:
                print(f"     {name:60} → requires_grad={p.requires_grad}")
                printed += 1
                if printed >= 5:
                    break