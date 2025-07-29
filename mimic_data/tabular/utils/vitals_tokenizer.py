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
        return batch