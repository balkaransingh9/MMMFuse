import torch
from .helper import compute_last_seen

class OutputTokenizer:
    def __init__(self, label_vocab, outnorm):
        self.label_vocab = label_vocab
        self.outnorm = outnorm

    def tokenize(self, out):
        B = len(out)
        L = max((len(x['label']) for x in out if x is not None), default=0)
        out_label_pad = torch.zeros(B, L).long()
        out_value_pad = torch.zeros(B, L)
        out_hours_pad = torch.zeros(B, L)
        out_hours_norm_pad = torch.zeros(B, L)
        out_last_seen_pad = torch.zeros(B, L)
        out_mask = torch.ones(B, L, dtype=torch.bool)

        for i, m in enumerate(out):
            if m is not None:
                labels = m['label']
                values = m['value']
                hours = m['hours_from_intime']
                l = len(labels)

                label_ids = [self.label_vocab.get(lbl, 0) for lbl in labels]

                out_label_pad[i, :l] = torch.tensor(label_ids)
                out_value_pad[i, :l] = (torch.tensor(values) - self.outnorm['value']['mean']) / self.outnorm['value']['std']
                out_hours_pad[i, :l] = torch.tensor(hours)
                out_hours_norm_pad[i, :l] = (out_hours_pad[i, :l] - self.outnorm['hours']['mean']) / self.outnorm['hours']['std']
                out_last_seen_pad[i, :l] = compute_last_seen(label_ids, hours)
                out_mask[i, :l] = False

        return {
            'hours': out_hours_pad,
            'hours_norm': out_hours_norm_pad,
            'value': out_value_pad,
            'label': out_label_pad,
            'last_seen': out_last_seen_pad,
            'mask': out_mask
        }