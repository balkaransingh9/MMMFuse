import torch

def make_param_groups(model, encoder_lrs, default_lr):
    """
    Build optimizer param_groups so that:
      - each encoder in model.encoders gets lr = encoder_lrs[name] if present,
        otherwise lr = default_lr
      - any other parameters get lr = default_lr
    """
    param_groups = []
    enc_params = set()

    # 1) one group per encoder, with either its special LR or default_lr
    for name, encoder in model.encoders.items():
        params = list(encoder.parameters())
        lr = encoder_lrs.get(name, default_lr)
        param_groups.append({
            "params": params,
            "lr":      lr
        })
        enc_params.update(params)

    # 2) the “rest” of the model’s parameters
    rest = [p for p in model.parameters() if p not in enc_params]
    if rest:
        param_groups.append({
            "params": rest,
            "lr":      default_lr
        })

    return param_groups
