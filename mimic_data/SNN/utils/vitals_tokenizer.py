import os
import json
import numpy as np
import torch

def F_impute(X, tt, mask, duration=1, tt_max=48):
    """
    Discretize into 1-hr bins and forward-fill (LOCF).

    Args:
        X   : (T, F) float32   values at unique timestamps
        tt  : (T,) float32     timestamps in hours in [0, tt_max]
        mask: (T, F) {0,1}     observation mask
        duration: bin width (hrs)
        tt_max : max time horizon (hrs)

    Returns:
        (tt_max // duration, F*2) float32
        [values_ffill | masks]
    """
    n_bins = tt_max // duration
    n_feat = X.shape[1]

    # last-observation-wins per bin
    temp_values = np.full((n_bins, n_feat), np.nan, dtype=np.float32)
    temp_masks  = np.zeros((n_bins, n_feat), dtype=np.float32)

    # place observations into bins
    for i in range(len(tt)):
        row = int(tt[i] // duration)
        if 0 <= row < n_bins:
            for f in range(n_feat):
                if mask[i, f] == 1:
                    temp_values[row, f] = X[i, f]
                    temp_masks[row, f]  = 1.0

    # forward fill; start from 0.0 if no prior value
    for f in range(n_feat):
        last = 0.0
        for r in range(n_bins):
            if not np.isnan(temp_values[r, f]):
                last = temp_values[r, f]
            else:
                temp_values[r, f] = last

    # concat values + masks
    out = np.zeros((n_bins, n_feat * 2), dtype=np.float32)
    out[:, :n_feat] = temp_values
    out[:, n_feat:] = temp_masks
    return out


class VitalTokenizer:
    """
    Build only the discretized, imputed grid: reg_ts (B, 48, F*2).

    Expected features (fixed order):
        0 Heart Rate
        1 Non Invasive Blood Pressure systolic
        2 Non Invasive Blood Pressure diastolic
        3 Non Invasive Blood Pressure mean
        4 Respiratory Rate
        5 O2 saturation pulseoxymetry
        6 GCS - Verbal Response
        7 GCS - Eye Opening
        8 GCS - Motor Response
        9 Temperature Celsius
    """
    def __init__(self, label_vocab=None, vitalnorm=None, channel_info_path=None):
        # vitalnorm['value'][label] -> {'mean': ..., 'std': ...}
        self.value_norm = (vitalnorm or {}).get('value', {})

        # Load GCS mappings
        if channel_info_path is None:
            try:
                here = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                here = os.getcwd()
            channel_info_path = os.path.join(here, 'channel_info.json')

        if not os.path.exists(channel_info_path):
            # fallback if file isn't available
            channel_info = {
                'Glascow coma scale verbal response': {'values': {}},
                'Glascow coma scale motor response':  {'values': {}},
                'Glascow coma scale eye opening':     {'values': {}},
            }
        else:
            with open(channel_info_path, 'r') as f:
                channel_info = json.load(f)

        self.gcs_mappings = {
            'GCS - Verbal Response':
                channel_info['Glascow coma scale verbal response']['values'],
            'GCS - Motor Response':
                channel_info['Glascow coma scale motor response']['values'],
            'GCS - Eye Opening':
                channel_info['Glascow coma scale eye opening']['values'],
        }

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
            'Temperature Celsius',
        ]
        
        self.feature_names = self.expected_cols + [f"{n}_mask" for n in self.expected_cols]
        self.n_features = len(self.expected_cols)
        self.label2idx = {lbl: i for i, lbl in enumerate(self.expected_cols)}
        self._numeric_keys = set(self.value_norm.keys())

    def tokenize(self, batch, tt_max=48, duration=1):
        """
        Args:
            batch: list of dicts with keys:
                'hours_from_intime' : list[float] timestamps in hours (0..48)
                'label'             : list[str]   same length as values
                'value'             : list[float|str]
        Returns:
            torch.FloatTensor of shape (B, tt_max//duration, F*2)
            (the reg_ts grid only)
        """
        B = len(batch)
        reg_list = []

        for samp in batch:
            hrs  = np.asarray(samp['hours_from_intime'], dtype=np.float32)
            labs = np.asarray(samp['label'],              dtype=object)
            vals = np.asarray(samp['value'],              dtype=object)

            # unique, sorted time points + inverse index per reading
            sorted_t, inv = np.unique(hrs, return_inverse=True)
            T = sorted_t.size

            # per-sample grids at unique times
            sample_vals  = np.zeros((T, self.n_features), dtype=np.float32)
            sample_masks = np.zeros((T, self.n_features), dtype=np.float32)

            # map labels to columns; drop unknown labels
            col_idx = np.array([self.label2idx.get(l, -1) for l in labs], dtype=np.int64)
            valid = col_idx >= 0
            col_idx = col_idx[valid]
            inv     = inv[valid]
            labs_v  = labs[valid]
            vals_v  = vals[valid]

            # numeric vs GCS categorical split
            is_num = np.array([l in self._numeric_keys for l in labs_v], dtype=bool)

            # numeric
            if np.any(is_num):
                num_idx = np.nonzero(is_num)[0]
                lbls_num = labs_v[num_idx]
                v_num    = vals_v[num_idx].astype(np.float32)
                means = np.array([self.value_norm[l]['mean'] for l in lbls_num], dtype=np.float32)
                stds  = np.array([self.value_norm[l]['std']  for l in lbls_num], dtype=np.float32)
                normed = (v_num - means) / (stds + 1e-8)

                r = inv[num_idx]
                c = col_idx[num_idx]
                sample_vals[r, c]  = normed
                sample_masks[r, c] = 1.0

            # categorical (GCS)
            if np.any(~is_num):
                cat_idx = np.nonzero(~is_num)[0]
                lbls_cat = labs_v[cat_idx]
                vals_cat = vals_v[cat_idx]
                mapped = np.zeros(cat_idx.size, dtype=np.float32)
                mmask  = np.zeros(cat_idx.size, dtype=np.float32)
                for j, (l, v) in enumerate(zip(lbls_cat, vals_cat)):
                    m = self.gcs_mappings.get(l, {}).get(str(v), None)
                    if m is not None:
                        mapped[j] = float(m)
                        mmask[j]  = 1.0

                r = inv[cat_idx]
                c = col_idx[cat_idx]
                sample_vals[r, c]  = mapped
                sample_masks[r, c] = mmask

            # discretize + forward fill into (48, F*2)
            reg_ts_sample = F_impute(
                X=sample_vals,
                tt=sorted_t,
                mask=sample_masks,
                duration=duration,
                tt_max=tt_max
            )
            reg_list.append(reg_ts_sample)

        reg_ts = np.stack(reg_list, axis=0)  # (B, 48, F*2)
        return {
            'values': torch.from_numpy(reg_ts),
            'feature_names': self.feature_names
        }