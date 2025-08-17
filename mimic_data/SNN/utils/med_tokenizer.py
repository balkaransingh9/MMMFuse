import numpy as np
import torch


def F_impute(X, tt, mask, duration=1, tt_max=48):
    """
    Discretize into duration-hour bins over [0, tt_max) and forward-fill (LOCF).

    Args:
        X   : (T, F) float32   values at unique timestamps
        tt  : (T,) float32     timestamps in hours (can be any float in [0, tt_max))
        mask: (T, F) {0,1}     observation mask
        duration: bin width (hrs)
        tt_max : max time horizon (hrs)

    Returns:
        (tt_max // duration, F*2) float32  -> [values_ffill | masks]
    """
    n_bins = tt_max // duration
    n_feat = X.shape[1]

    temp_values = np.full((n_bins, n_feat), np.nan, dtype=np.float32)
    temp_masks  = np.zeros((n_bins, n_feat), dtype=np.float32)

    # Last observation wins inside each bin
    for i in range(len(tt)):
        row = int(tt[i] // duration)
        if 0 <= row < n_bins:
            # write all features observed at this time index
            for f in range(n_feat):
                if mask[i, f] == 1:
                    temp_values[row, f] = X[i, f]
                    temp_masks[row, f]  = 1.0

    # Forward fill (start from 0.0 if no prior observation)
    for f in range(n_feat):
        last = 0.0
        for r in range(n_bins):
            if not np.isnan(temp_values[r, f]):
                last = temp_values[r, f]
            else:
                temp_values[r, f] = last

    out = np.zeros((n_bins, n_feat * 2), dtype=np.float32)
    out[:, :n_feat] = temp_values
    out[:, n_feat:] = temp_masks
    return out


class MedTokenizer:
    """
    Produces an hourly-binned, forward-filled table with labels as columns.

    Output:
        {
          'reg_ts':  FloatTensor (B, 48, F*2)  # first F = values, last F = masks
          'columns': List[str]                 # label order for the first F cols
        }
    """
    def __init__(self, label_vocab, unit_vocab=None, cat_vocab=None, mednorm=None,
                 duration=1, tt_max=48):
        # Fix a stable column order from label_vocab ids (ascending)
        # label_vocab: dict[str -> int]
        self.columns = [lbl for lbl, _ in sorted(label_vocab.items(), key=lambda kv: kv[1])]
        self.label2idx = {lbl: i for i, lbl in enumerate(self.columns)}
        self.n_features = len(self.columns)

        # params for binning
        self.duration = duration
        self.tt_max = tt_max

        # (We ignore unit/category as requested; mednorm not used here.)

    def tokenize(self, meds):
        """
        meds: list of dicts (one per sample), each with:
            'label'              : list[str]
            'amount_std_value'   : list[float]
            'hours_from_intime'  : list[float]

        Returns:
            dict with:
              'reg_ts':  torch.FloatTensor (B, tt_max//duration, F*2)
              'columns': list[str] of length F
        """
        B = len(meds)
        n_bins = self.tt_max // self.duration
        reg_list = []

        for m in meds:
            # Handle empty / None sample
            if not m or len(m.get('label', [])) == 0:
                reg_list.append(np.zeros((n_bins, self.n_features * 2), dtype=np.float32))
                continue

            labs = np.asarray(m['label'], dtype=object)
            vals = np.asarray(m['amount_std_value'], dtype=np.float32)
            hrs  = np.asarray(m['hours_from_intime'], dtype=np.float32)

            # Keep only known labels
            valid = np.array([lbl in self.label2idx for lbl in labs], dtype=bool)
            labs = labs[valid]
            vals = vals[valid]
            hrs  = hrs[valid]

            if labs.size == 0:
                reg_list.append(np.zeros((n_bins, self.n_features * 2), dtype=np.float32))
                continue

            # Unique sorted times (per sample) + inverse index for each original reading
            sorted_t, inv = np.unique(hrs, return_inverse=True)
            T = sorted_t.size

            # Wide per-sample table at unique times
            sample_vals  = np.zeros((T, self.n_features), dtype=np.float32)
            sample_masks = np.zeros((T, self.n_features), dtype=np.float32)

            # Fill table; iterate in original order so "last write wins" at same timestamp
            for j in range(labs.size):
                r = inv[j]
                c = self.label2idx[labs[j]]
                sample_vals[r, c]  = vals[j]
                sample_masks[r, c] = 1.0

            # Hourly bins + forward-fill imputation
            reg_ts_sample = F_impute(
                X=sample_vals,
                tt=sorted_t,
                mask=sample_masks,
                duration=self.duration,
                tt_max=self.tt_max
            )
            reg_list.append(reg_ts_sample)

        reg_ts = np.stack(reg_list, axis=0)  # (B, n_bins, F*2)
        return {
            'reg_ts': torch.from_numpy(reg_ts),
            'columns': self.columns,
        }
