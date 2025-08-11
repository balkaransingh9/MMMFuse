import torch
from .helper import compute_last_seen

class MedTokenizer:
    def __init__(self, label_vocab, unit_vocab, cat_vocab, mednorm):
        self.label_vocab = label_vocab
        self.unit_vocab = unit_vocab
        self.cat_vocab = cat_vocab
        self.mednorm = mednorm

    def tokenize(self, meds):
        B = len(meds)
        L = max((len(x['label']) for x in meds if x is not None), default=0)
        med_id_pad = torch.zeros(B, L).long()
        med_unit_pad = torch.zeros(B, L).long()
        med_cat_pad = torch.zeros(B, L).long()
        med_value_pad = torch.zeros(B, L)
        med_hours_pad = torch.zeros(B, L)
        med_hours_norm_pad = torch.zeros(B, L)
        med_last_seen_pad = torch.zeros(B, L)
        med_mask = torch.ones(B, L, dtype=torch.bool)

        for i, m in enumerate(meds):
            if m is not None:
                labels = m['label']
                units = m['amount_std_uom']
                cats = m['ordercategoryname']
                values = m['amount_std_value']
                hours = m['hours_from_intime']
                l = len(labels)

                label_ids = [self.label_vocab.get(lbl, 0) for lbl in labels]
                unit_ids = [self.unit_vocab.get(u, 0) for u in units]
                cat_ids = [self.cat_vocab.get(c, 0) for c in cats]

                med_id_pad[i, :l] = torch.tensor(label_ids)
                med_unit_pad[i, :l] = torch.tensor(unit_ids)
                med_cat_pad[i, :l] = torch.tensor(cat_ids)
                med_value_pad[i, :l] = (torch.tensor(values) - self.mednorm['value']['mean']) / self.mednorm['value']['std']
                med_hours_pad[i, :l] = torch.tensor(hours)
                med_hours_norm_pad[i, :l] = (med_hours_pad[i, :l] - self.mednorm['hours']['mean']) / self.mednorm['hours']['std']
                med_last_seen_pad[i, :l] = compute_last_seen(label_ids, hours)
                med_mask[i, :l] = False

        return {
            'hours': med_hours_pad,
            'hours_norm': med_hours_norm_pad,
            'label': med_id_pad,
            'value': med_value_pad,
            'unit': med_unit_pad,
            'category': med_cat_pad,
            'last_seen': med_last_seen_pad,
            'mask': med_mask
        }