import torch

class LabTokenizer:
    def __init__(self, label_vocab, labnorm):
        self.label_vocab = label_vocab
        self.labnorm = labnorm

    def tokenize(self, meds):
        B = len(meds)
        L = max(len(x['label']) for x in meds if x is not None)
        lab_label_pad = torch.zeros(B, L).long()
        lab_value_pad = torch.zeros(B, L)
        lab_hours_pad = torch.zeros(B, L)
        lab_mask = torch.zeros(B, L, dtype=torch.bool)

        for i, m in enumerate(meds):
            if m is not None:
                l = len(m['label'])
                lab_label_pad[i, :l] = torch.tensor([self.label_vocab.get(i) for i in m['label']])
                lab_value_pad[i, :l] = torch.tensor((m['value'] - self.labnorm['value']['mean']) / self.labnorm['value']['std'])
                lab_hours_pad[i, :l] = torch.tensor((m['hours_from_intime'] - self.labnorm['hours']['mean']) / self.labnorm['hours']['std'])
                lab_mask[i, l:] = True
            else:
                lab_mask[i, :] = True

        return {
            'hours': lab_hours_pad,
            'label': lab_label_pad,
            'value': lab_value_pad,
        }