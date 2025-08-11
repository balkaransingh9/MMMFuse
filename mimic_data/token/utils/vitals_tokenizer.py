import torch

class VitalTokenizer:
    def __init__(self, label_vocab, vitalnorm, 
                 discrete_label_categorical_values=None, pad_idx=0):
        self.vital_label_vocab = label_vocab

        self.vital_value_normaliser = vitalnorm['value']

        self.tokenizer_vocab = {
            "<PAD>": 0,       # Padding token
            "<UNK>": 1,       # Unknown token

            "No Response-ETT": 2,
            "Incomprehensible sounds": 3,
            "Inappropriate Words": 4,
            "No Response": 5,
            "Confused": 6,
            "Oriented": 7,

            "Abnormal Flexion": 8,
            "Obeys Commands": 9,
            "No response": 10,
            "Abnormal extension": 11,
            "Flex-withdraws": 12,
            "Localizes Pain": 13,

            "no-opening": 14,
            "To Pain": 15,
            "Spontaneously": 16,
            "To Speech": 17
            }

        self.categoricals = ['GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response']

    def tokenize(self, batch):
        B = len(batch)
        L = max((len(x['label']) for x in batch if x is not None), default=0)
        discrete_label_values = self.tokenizer_vocab.keys()

        labels_pad = torch.zeros((B, L), dtype=torch.long)
        values_pad = torch.zeros((B, L), dtype=torch.float32)
        value_id_pad = torch.zeros((B, L), dtype=torch.long)
        hours_pad = torch.zeros((B, L), dtype=torch.float32)
        mask_pad = torch.ones((B, L), dtype=torch.bool)

        min_hours = 0
        max_hours = 48

        for i, m in enumerate(batch):
            if m is not None:
                labels = m['label']
                values = m['value']
                hours = m['hours_from_intime']
                l = len(labels)

                label_ids = [self.vital_label_vocab.get(lbl, 0) for lbl in labels]
                labels_pad[i, :l] = torch.tensor(label_ids)
                value_id_pad[i, :l] = torch.tensor([self.tokenizer_vocab.get(v, self.tokenizer_vocab['<UNK>']) for v in values])
                
                normalized_hours = [hour / max_hours if max_hours > min_hours else 0.0 for hour in hours]
                hours_pad[i, :l] = torch.tensor(normalized_hours)
                mask_pad[i, :l] = False

                for j, l in enumerate(labels):
                    if l in self.categoricals:
                        values_pad[i, j] = 0   # For categorical we set values to 0
                    else:
                        stats = self.vital_value_normaliser.get(l)
                        if stats:
                            mean = stats['mean']
                            std = stats['std']
                            values_pad[i, j] = (float(values[j]) - mean) / (std + 1e-6)
                        else:
                            values_pad[i, j] = 0.0
        
        return {
            'label': labels_pad,
            'value': values_pad,
            'value_id': value_id_pad,
            'hours': hours_pad,
            'mask':  mask_pad
        }