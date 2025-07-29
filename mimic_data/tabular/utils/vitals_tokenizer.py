import torch
import json
import pandas as pd
import numpy as np
import os

class VitalTokenizer:
    def __init__(self, label_vocab, vitalnorm):
        self.label_vocab = label_vocab

        self.value_norm = vitalnorm['value']
        self.time_norm = vitalnorm['hours']  # Not used, but included for completeness

        # Always find JSON files relative to this Python file
        here = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(here, 'channel_info.json'), 'r') as f:
            self.channel_info = json.load(f)

        with open(os.path.join(here, 'discretizer_config.json'), 'r') as f:
            self.discretizer_config = json.load(f)

    def tokenize(self, batch):
        time_list = []
        values_list = []
        masks_list = []

        maxlen = 0
        for samp_dict in batch:
            df = pd.DataFrame(samp_dict)
            pivoted = df.pivot(index='hours_from_intime', columns='label', values='value')
            pivoted = pivoted.reset_index()

            # Normalize values (z-score)
            for col in pivoted.columns:
                if col in self.value_norm:
                    pivoted[col] = pd.to_numeric(pivoted[col], errors='coerce')
                    mean = self.value_norm[col]['mean']
                    std = self.value_norm[col]['std']
                    pivoted[col] = (pivoted[col] - mean) / std

            # Min-max normalization of time: min=0, max=48
            t = pivoted['hours_from_intime'].to_numpy()
            t = (t - 0.0) / (48.0 - 0.0 + 1e-8)

            # Map categoricals (only if column present)
            if 'GCS - Verbal Response' in pivoted:
                pivoted['GCS - Verbal Response'] = pivoted['GCS - Verbal Response'].map(
                    self.channel_info['Glascow coma scale verbal response']['values'])
            if 'GCS - Motor Response' in pivoted:
                pivoted['GCS - Motor Response'] = pivoted['GCS - Motor Response'].map(
                    self.channel_info['Glascow coma scale motor response']['values'])
            if 'GCS - Eye Opening' in pivoted:
                pivoted['GCS - Eye Opening'] = pivoted['GCS - Eye Opening'].map(
                    self.channel_info['Glascow coma scale eye opening']['values'])

            # Mask: 1 if not NaN, else 0 (drop time col)
            mask = ~pivoted.isna()
            mask = mask.astype(int).drop(columns=['hours_from_intime']).to_numpy()
            # Fill missing values with 0 for data
            values = pivoted.drop(columns=['hours_from_intime']).fillna(0).to_numpy()

            seqlen = len(t)
            maxlen = max(maxlen, seqlen)

            masks_list.append(mask)
            values_list.append(values)
            time_list.append(t)

        n_features = values_list[0].shape[1]

        padded_times = []
        padded_values = []
        padded_masks = []
        for t, v, m in zip(time_list, values_list, masks_list):
            pad_len = maxlen - len(t)
            # Pad time with zeros
            padded_t = np.pad(t, (0, pad_len), 'constant', constant_values=0)
            # Pad values with zeros
            padded_v = np.pad(v, ((0, pad_len), (0,0)), 'constant', constant_values=0)
            # Pad mask with zeros
            padded_m = np.pad(m, ((0, pad_len), (0,0)), 'constant', constant_values=0)
            padded_times.append(padded_t)
            padded_values.append(padded_v)
            padded_masks.append(padded_m)

        time = torch.tensor(padded_times, dtype=torch.float32)
        values = torch.tensor(padded_values, dtype=torch.float32)
        masks = torch.tensor(padded_masks, dtype=torch.int64)
        return {'time': time, 'values': values, 'masks': masks}