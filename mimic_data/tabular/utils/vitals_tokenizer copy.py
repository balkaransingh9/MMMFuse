import os
import json
import numpy as np
import torch

class VitalTokenizer:
    def __init__(self, label_vocab, vitalnorm):
        self.label_vocab = label_vocab
        self.value_norm = vitalnorm['value']

        # load channel info (for GCS mappings) and discretizer config
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, 'channel_info.json'), 'r') as f:
            channel_info = json.load(f)

        # build maps for the three GCS features
        self.gcs_mappings = {
            'GCS - Verbal Response':
                channel_info['Glascow coma scale verbal response']['values'],
            'GCS - Motor Response':
                channel_info['Glascow coma scale motor response']['values'],
            'GCS - Eye Opening':
                channel_info['Glascow coma scale eye opening']['values'],
        }

        # the 10 expected features, in fixed order
        self.expected_cols = [
            'Heart Rate',
            'Non Invasive Blood Pressure systolic',
            'Non Invasive Blood Pressure diastolic',
            'Non Invasive Blood Pressure mean',
            'Respiratory Rate',
            'O2 saturation pulseoxymetry',
            'GCS - Verbal Response',
            'GCS - Eye Opening',
            'GCS - Motor Response',
            'Temperature Celsius'
        ]
        self.n_features = len(self.expected_cols)
        # quick label → column index
        self.label2idx = {lbl: i for i, lbl in enumerate(self.expected_cols)}

    def tokenize(self, batch):
        """
        batch: list of dicts, each with keys 'hours_from_intime', 'label', 'value'
        returns dict with:
          time:   (B, T) float32 tensor in [0,1]
          values: (B, T, F) float32 tensor
          masks:  (B, T, F) int64 tensor (1 if present, 0 if padded/missing)
        """
        B = len(batch)
        # first pass to find maximum unique time‐steps
        maxlen = 0
        for samp in batch:
            unique = np.unique(samp['hours_from_intime'])
            if unique.size > maxlen:
                maxlen = unique.size

        # preallocate
        times  = np.zeros((B, maxlen), dtype=np.float32)
        vals   = np.zeros((B, maxlen, self.n_features), dtype=np.float32)
        masks  = np.zeros((B, maxlen, self.n_features), dtype=np.int64)

        for i, samp in enumerate(batch):
            hrs   = np.array(samp['hours_from_intime'], dtype=np.float32)
            labs  = np.array(samp['label'], dtype=object)
            vraw  = np.array(samp['value'], dtype=object)

            # get unique sorted times and an inverse index for each reading
            uniq, inv = np.unique(hrs, return_inverse=True)
            order = np.argsort(uniq)
            inv   = order[inv]
            sorted_t = uniq[order]

            T = sorted_t.size
            # normalize time into [0,1] over 0–48 hours
            times[i, :T] = (sorted_t - 0.0) / (48.0 + 1e-8)

            # map labels → feature columns
            col_idx = np.array([ self.label2idx.get(l, -1) for l in labs ], dtype=np.int64)
            valid   = col_idx >= 0
            col_idx = col_idx[valid]
            inv     = inv[valid]
            labs    = labs[valid]
            vraw    = vraw[valid]

            # split into numeric vs GCS‐categorical
            is_numeric = np.array([lbl in self.value_norm for lbl in labs])
            # --- numeric portion ---
            if is_numeric.any():
                num_idx   = np.nonzero(is_numeric)[0]
                lbls_num  = labs[num_idx]
                vals_num  = vraw[num_idx].astype(np.float32)
                means     = np.array([ self.value_norm[l]['mean'] for l in lbls_num ], dtype=np.float32)
                stds      = np.array([ self.value_norm[l]['std']  for l in lbls_num ], dtype=np.float32)
                normed    = (vals_num - means) / stds

                r = inv[num_idx]
                c = col_idx[num_idx]
                vals[i, r, c]  = normed
                masks[i, r, c] = 1

            # --- categorical (GCS) portion ---
            cat_idx = np.nonzero(~is_numeric)[0]
            if cat_idx.size > 0:
                lbls_cat = labs[cat_idx]
                vals_cat = vraw[cat_idx]
                mapped   = np.zeros(cat_idx.size, dtype=np.float32)
                mmask    = np.zeros(cat_idx.size, dtype=np.int64)
                for j, (l, v) in enumerate(zip(lbls_cat, vals_cat)):
                    m = self.gcs_mappings[l].get(str(v), None)
                    if m is not None:
                        mapped[j] = m
                        mmask[j]  = 1

                r = inv[cat_idx]
                c = col_idx[cat_idx]
                vals[i, r, c]  = mapped
                masks[i, r, c] = mmask

        # convert to torch
        return {
            'time':   torch.from_numpy(times),
            'values': torch.from_numpy(vals),
            'masks':  torch.from_numpy(masks),
        }