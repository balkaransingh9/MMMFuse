import torch
from .helper import compute_last_seen

class VitalTokenizer:
    def __init__(self, label_vocab, vitalnorm, discrete_label_categorical_values, pad_idx=0):
        self.label_vocab = label_vocab
        self.pad_idx = pad_idx

        self.val_stats = vitalnorm['value']
        self.hrs_mean = vitalnorm['hours']['mean']
        self.hrs_std = vitalnorm['hours']['std']

        self.discrete_labels = set(discrete_label_categorical_values.keys())
        self.val2id = {
            lbl: {cat: i for i, cat in enumerate(cats)} | {'<UNK>': len(cats)}
            for lbl, cats in discrete_label_categorical_values.items()
        }

    def tokenize(self, batch):
        B = len(batch)
        L = max((len(x['label']) if x else 0) for x in batch)

        labels_pad = torch.full((B, L), self.pad_idx, dtype=torch.long)
        values_pad = torch.zeros((B, L), dtype=torch.float32)
        vid_pad = torch.zeros((B, L), dtype=torch.long)
        hours_pad = torch.zeros((B, L), dtype=torch.float32)
        hours_norm_pad = torch.zeros((B, L), dtype=torch.float32)
        last_seen_pad = torch.zeros((B, L), dtype=torch.float32)
        mask_pad = torch.ones((B, L), dtype=torch.bool)

        for i, rec in enumerate(batch):
            if not rec:
                continue

            labels = rec['label']
            values = rec.get('value', [])
            hours = rec.get('hours_from_intime', [])
            length = len(labels)
            mask_pad[i, :length] = False

            label_ids = [self.label_vocab.get(l, self.pad_idx) for l in labels]
            labels_pad[i, :length] = torch.tensor(label_ids)

            hours_pad[i, :length] = torch.tensor(hours, dtype=torch.float32)
            hours_norm_pad[i, :length] = (hours_pad[i, :length] - self.hrs_mean) / (self.hrs_std + 1e-6)
            last_seen_pad[i, :length] = compute_last_seen(label_ids, hours)

            for j, l in enumerate(labels):
                if l in self.discrete_labels:
                    vid_pad[i, j] = self.val2id[l].get(values[j], self.val2id[l]['<UNK>'])
                else:
                    stats = self.val_stats.get(l)
                    if stats:
                        mean = stats['mean']
                        std = stats['std']
                        values_pad[i, j] = (float(values[j]) - mean) / (std + 1e-6)
                    else:
                        values_pad[i, j] = 0.0

        return {
            'label': labels_pad,
            'value': values_pad,
            'value_id': vid_pad,
            'hours': hours_pad,
            'hours_norm': hours_norm_pad,
            'last_seen': last_seen_pad,
            'mask': mask_pad
        }