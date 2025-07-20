import torch

class MedTokenizer:
    def __init__(self, label_vocab, unit_vocab, cat_vocab, mednorm):
        self.label_vocab = label_vocab
        self.unit_vocab = unit_vocab
        self.cat_vocab = cat_vocab
        self.mednorm = mednorm

    def tokenize(self, meds):
        B = len(meds)
        L = max(len(x['label']) for x in meds if x is not None)
        med_id_pad = torch.zeros(B, L).long()
        med_unit_pad = torch.zeros(B, L).long()
        med_cat_pad = torch.zeros(B, L).long()
        med_value_pad = torch.zeros(B, L)
        med_hours_pad = torch.zeros(B, L)
        med_mask = torch.zeros(B, L, dtype=torch.bool)

        for i, m in enumerate(meds):
            if m is not None:
                l = len(m['label'])
                med_id_pad[i, :l] = torch.tensor([self.label_vocab.get(i) for i in m['label']])
                med_unit_pad[i, :l] = torch.tensor([self.unit_vocab.get(i) for i in m['amount_std_uom']])
                med_cat_pad[i, :l] = torch.tensor([self.cat_vocab.get(i) for i in m['ordercategoryname']])
                med_value_pad[i, :l] = torch.tensor((m['amount_std_value'] - self.mednorm['value']['mean']) / self.mednorm['value']['std'])
                med_hours_pad[i, :l] = torch.tensor((m['hours_from_intime'] - self.mednorm['hours']['mean']) / self.mednorm['hours']['std'])
                med_mask[i, l:] = True
            else:
                med_mask[i, :] = True

        return {
            'hours': med_hours_pad,
            'label': med_id_pad,
            'value': med_value_pad,
            'unit': med_unit_pad,
            'category': med_cat_pad,
            'mask': med_mask
        }