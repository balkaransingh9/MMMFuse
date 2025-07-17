import torch

class VitalTokenizer:
    def __init__(self, label_vocab, vitalnorm):
        self.label_vocab = label_vocab
        self.vitalnorm = vitalnorm

    def tokenize(self, meds):
        B = len(meds)
        L = max(len(x['label']) for x in meds if x is not None)
        vitals_label_pad = torch.zeros(B, L).long()
        vitals_value_pad = torch.zeros(B, L)
        vitals_hours_pad = torch.zeros(B, L)
        vitals_mask = torch.zeros(B, L, dtype=torch.bool)

        for i, m in enumerate(meds):
            if m is not None:
                l = len(m['label'])
                vitals_label_pad[i, :l] = torch.tensor([self.label_vocab.get(i) for i in m['label']])
                vitals_value_pad[i, :l] = torch.tensor((m['amount_std_value'] - self.vitalnorm['value']['mean']) / self.vitalnorm['value']['std'])
                vitals_hours_pad[i, :l] = torch.tensor((m['hours_from_intime'] - self.vitalnorm['hours']['mean']) / self.vitalnorm['hours']['std'])
                vitals_mask[i, l:] = True
            else:
                vitals_mask[i, :] = True

        return {
            'hours': vitals_hours_pad,
            'label': vitals_label_pad,
            'value': vitals_value_pad,
        }