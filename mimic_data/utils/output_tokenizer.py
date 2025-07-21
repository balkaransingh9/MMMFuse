import torch

class OutputTokenizer:
    def __init__(self, label_vocab, outnorm):
        self.label_vocab = label_vocab
        self.outnorm = outnorm

    def tokenize(self, out):
        B = len(out)
        L = max(len(x['label']) for x in out if x is not None)
        out_label_pad = torch.zeros(B, L).long()
        out_value_pad = torch.zeros(B, L)
        out_hours_pad = torch.zeros(B, L)
        out_hours_norm_pad = torch.zeros(B, L)
        out_mask = torch.zeros(B, L, dtype=torch.bool)

        for i, m in enumerate(out):
            if m is not None:
                l = len(m['label'])
                out_label_pad[i, :l] = torch.tensor([self.label_vocab.get(i) for i in m['label']])
                out_value_pad[i, :l] = torch.tensor((m['value'] - self.outnorm['value']['mean']) / self.outnorm['value']['std'])
                out_hours_norm_pad[i, :l] = torch.tensor((m['hours_from_intime'] - self.outnorm['hours']['mean']) / self.outnorm['hours']['std'])
                out_hours_pad[i, :l] = m['hours_from_intime']
                out_mask[i, l:] = True
            else:
                out_mask[i, :] = True

        return {
            'hours': out_hours_pad,
            'hours_norm': out_hours_norm_pad,
            'value': out_value_pad,
            'label': out_label_pad,
            'mask': out_mask
        }