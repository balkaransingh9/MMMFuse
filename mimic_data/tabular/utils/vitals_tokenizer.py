import os
import json
import numpy as np
import torch

def F_impute(X, tt, mask, duration, tt_max):
    """
    Performs discretization and forward-fill imputation on a single time series instance.

    Args:
        X (np.array): Feature values, shape (T, F).
        tt (np.array): Timestamps for each observation, shape (T,). Assumed to be scaled to the [0, tt_max] range.
        mask (np.array): Observation mask, shape (T, F).
        duration (int): The duration of each discrete time bin.
        tt_max (int): The maximum timestamp value, defining the total number of bins.

    Returns:
        np.array: A discretized and imputed time series grid of shape (tt_max // duration, F * 2).
    """
    no_feature = X.shape[1]
    # Create the regular grid, initialized to zeros.
    impute = np.zeros(shape=(tt_max // duration, no_feature * 2), dtype=np.float32)

    # Create a temporary grid to handle multiple observations per bin and perform forward fill later.
    temp_values = np.full((tt_max // duration, no_feature), np.nan, dtype=np.float32)
    temp_masks = np.zeros((tt_max // duration, no_feature), dtype=np.float32)

    # --- Pass 1: Place all observed values into the temporary grid ---
    # This handles cases where multiple observations fall into the same time bin by keeping the last one.
    for i in range(len(tt)):
        # Skip if timestamp is invalid (e.g., padding)
        if tt[i] == 0 and i > 0 and np.all(tt[:i] != 0): # A simple check for padding time
             continue

        row = int(tt[i] / duration)
        if row >= (tt_max // duration):
            continue

        for f_idx in range(no_feature):
            if mask[i, f_idx] == 1:
                temp_values[row, f_idx] = X[i, f_idx]
                temp_masks[row, f_idx] = 1


    # --- Pass 2: Perform forward-fill (LOCF) on the temporary grid ---
    for f_idx in range(no_feature):
        last_observed_val = 0.0 # Start with 0 for initial missing values
        for row in range(tt_max // duration):
            if not np.isnan(temp_values[row, f_idx]):
                last_observed_val = temp_values[row, f_idx]
            else:
                temp_values[row, f_idx] = last_observed_val

    # --- Pass 3: Populate the final output matrix ---
    impute[:, :no_feature] = temp_values
    impute[:, no_feature:] = temp_masks

    return impute


class VitalTokenizer:
    def __init__(self, label_vocab, vitalnorm):
        self.label_vocab = label_vocab
        self.value_norm = vitalnorm['value']

        # load channel info (for GCS mappings) and discretizer config
        here = os.path.dirname(os.path.abspath(__file__))
        # Make sure the 'channel_info.json' file is in the same directory as this script
        channel_info_path = os.path.join(here, 'channel_info.json')
        if not os.path.exists(channel_info_path):
             # A fallback for environments where file access might be restricted
             print(f"Warning: 'channel_info.json' not found at {channel_info_path}. GCS mapping will be empty.")
             channel_info = {
                  'Glascow coma scale verbal response': {'values': {}},
                  'Glascow coma scale motor response': {'values': {}},
                  'Glascow coma scale eye opening': {'values': {}}
             }
        else:
            with open(channel_info_path, 'r') as f:
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
          reg_ts: (B, S, F*2) float32 tensor (discretized to S=48 steps)
        """
        B = len(batch)
        # first pass to find maximum unique time‐steps
        maxlen = 0
        for samp in batch:
            unique = np.unique(samp['hours_from_intime'])
            if unique.size > maxlen:
                maxlen = unique.size

        # preallocate for irregular data
        times = np.zeros((B, maxlen), dtype=np.float32)
        vals = np.zeros((B, maxlen, self.n_features), dtype=np.float32)
        masks = np.zeros((B, maxlen, self.n_features), dtype=np.int64)

        # preallocate for regularized (discretized) data
        reg_ts_batch = np.zeros((B, 48, self.n_features * 2), dtype=np.float32)


        for i, samp in enumerate(batch):
            hrs = np.array(samp['hours_from_intime'], dtype=np.float32)
            labs = np.array(samp['label'], dtype=object)
            vraw = np.array(samp['value'], dtype=object)

            # get unique sorted times and an inverse index for each reading
            uniq, inv = np.unique(hrs, return_inverse=True)
            order = np.argsort(uniq)
            inv = order[inv]
            sorted_t = uniq[order]

            T = sorted_t.size
            # normalize time into [0,1] over 0–48 hours
            times[i, :T] = (sorted_t - 0.0) / (48.0 + 1e-8)

            # map labels → feature columns
            col_idx = np.array([ self.label2idx.get(l, -1) for l in labs ], dtype=np.int64)
            valid = col_idx >= 0
            col_idx = col_idx[valid]
            inv = inv[valid]
            labs = labs[valid]
            vraw = vraw[valid]

            # split into numeric vs GCS‐categorical
            is_numeric = np.array([lbl in self.value_norm for lbl in labs])
            # --- numeric portion ---
            if is_numeric.any():
                num_idx = np.nonzero(is_numeric)[0]
                lbls_num = labs[num_idx]
                vals_num = vraw[num_idx].astype(np.float32)
                means = np.array([ self.value_norm[l]['mean'] for l in lbls_num ], dtype=np.float32)
                stds = np.array([ self.value_norm[l]['std']  for l in lbls_num ], dtype=np.float32)
                normed = (vals_num - means) / (stds + 1e-8) # Added epsilon for safety

                r = inv[num_idx]
                c = col_idx[num_idx]
                vals[i, r, c] = normed
                masks[i, r, c] = 1

            # --- categorical (GCS) portion ---
            cat_idx = np.nonzero(~is_numeric)[0]
            if cat_idx.size > 0:
                lbls_cat = labs[cat_idx]
                vals_cat = vraw[cat_idx]
                mapped = np.zeros(cat_idx.size, dtype=np.float32)
                mmask = np.zeros(cat_idx.size, dtype=np.int64)
                for j, (l, v) in enumerate(zip(lbls_cat, vals_cat)):
                    m = self.gcs_mappings[l].get(str(v), None)
                    if m is not None:
                        mapped[j] = m
                        mmask[j] = 1

                r = inv[cat_idx]
                c = col_idx[cat_idx]
                vals[i, r, c] = mapped
                masks[i, r, c] = mmask

            # --- NEW: Create the discretized version for this sample ---
            # Use the processed values and masks for the current sample `i`
            sample_vals = vals[i, :T, :]
            sample_masks = masks[i, :T, :]
            # Un-normalize the timestamps to be in the [0, 48] range for F_impute
            sample_times_scaled = sorted_t

            # Call the imputation function
            reg_ts_sample = F_impute(
                X=sample_vals,
                tt=sample_times_scaled,
                mask=sample_masks,
                duration=1,
                tt_max=48
            )
            reg_ts_batch[i] = reg_ts_sample


        # convert to torch
        return {
            'time':   torch.from_numpy(times),
            'values': torch.from_numpy(vals),
            'masks':  torch.from_numpy(masks),
            'reg_ts': torch.from_numpy(reg_ts_batch) # Add the new discretized data
        }