import numpy as np
import torch


def _forward_fill(values_2d: np.ndarray, start_fill: float = 0.0) -> np.ndarray:
    """
    Forward-fill each column independently. Leading NaNs are replaced with start_fill.
    values_2d: (R, C) with NaNs where unknown.
    """
    R, C = values_2d.shape
    out = np.empty_like(values_2d, dtype=np.float32)

    for c in range(C):
        col = values_2d[:, c]
        not_nan = ~np.isnan(col)
        # last valid index up to each row; -1 where none yet
        last_idx = np.where(not_nan, np.arange(R), -1)
        last_idx = np.maximum.accumulate(last_idx)

        filled = np.where(last_idx >= 0, col[last_idx], np.float32(start_fill))
        out[:, c] = filled

    return out


class MedTokenizer:
    """
    Tabular/discretized medication grid (similar to your vitals reg_ts).

    For each sample:
      - Time is binned into duration-hour bins over [0, tt_max].
      - Within each bin and medication label, the last observed value wins.
      - Then we forward-fill across time (LOCF).
      - Output per sample is shape (n_bins, 2*F): [values_ffill | masks].

    Inputs per sample dict `m` (same keys you already have):
      m['label']              : list[str]   medication names
      m['amount_std_value']   : list[float] standardised numeric amounts
      m['hours_from_intime']  : list[float] timestamps in hours (0..tt_max)

    Args to __init__:
      label_vocab : dict[str,int]  vocabulary of medication labels
      unit_vocab  : (ignored)
      cat_vocab   : (ignored)
      mednorm     : dict with:
                    mednorm['value']['mean'], mednorm['value']['std'] (scalars)

    Returns from tokenize(meds, tt_max=48, duration=1):
      {
        'values': FloatTensor of shape (B, n_bins, 2*F)
      }
    """
    def __init__(self, label_vocab, unit_vocab=None, cat_vocab=None, mednorm=None):
        self.label_vocab = label_vocab or {}
        self.mednorm = mednorm or {'value': {'mean': 0.0, 'std': 1.0}}

        # Build a stable column order from label_vocab indices if they look contiguous,
        # otherwise just sort by label name.
        try:
            items = sorted(self.label_vocab.items(), key=lambda kv: int(kv[1]))
        except Exception:
            items = sorted(self.label_vocab.items(), key=lambda kv: kv[0])

        self.labels_ordered = [k for k, _ in items]
        self.label2col = {lbl: i for i, lbl in enumerate(self.labels_ordered)}
        self.n_features = len(self.labels_ordered)

        self.feature_names = self.labels_ordered + [f"{n} (Missing Mask)" for n in self.labels_ordered]

        self._val_mean = float(self.mednorm.get('value', {}).get('mean', 0.0))
        self._val_std  = float(self.mednorm.get('value', {}).get('std',  1.0))

    def tokenize(self, meds, tt_max: int = 48, duration: int = 1):
        """
        Args:
          meds: list[dict|None]
          tt_max: max hours (inclusive end clamped into last bin)
          duration: bin width in hours (e.g., 1 -> hourly bins)

        Returns:
          {'values': torch.FloatTensor(B, n_bins, 2*F)}
        """
        B = len(meds)
        F = self.n_features
        n_bins = int(tt_max // duration)

        # Prepare output container
        batch_out = np.zeros((B, n_bins, 2 * F), dtype=np.float32)

        for i, m in enumerate(meds):
            if not m:
                # No meds for this sample → zeros (values=0 via start_fill, masks=0)
                continue

            labels = m.get('label', [])
            vals   = m.get('amount_std_value', [])
            hours  = m.get('hours_from_intime', [])

            if len(labels) == 0:
                continue

            labels = np.asarray(labels, dtype=object)
            vals   = np.asarray(vals,   dtype=np.float32)
            hours  = np.asarray(hours,  dtype=np.float32)

            # normalize values (scalar mean/std as in your current tokenizer)
            vals = (vals - self._val_mean) / (self._val_std + 1e-8)

            # Map to columns; drop unknown labels
            cols = np.array(
                [self.label2col[l] for l in labels if l in self.label2col],
                dtype=np.int64
            )
            keep_mask = np.array([l in self.label2col for l in labels], dtype=bool)
            if not keep_mask.any():
                continue

            vals  = vals[keep_mask]
            hours = hours[keep_mask]

            # Bin rows; clamp tt == tt_max into the last bin
            rows = np.floor(hours / duration).astype(np.int64)
            rows = np.clip(rows, 0, n_bins - 1)

            # last-observation-wins per (row, col)
            temp_values = np.full((n_bins, F), np.nan, dtype=np.float32)
            temp_masks  = np.zeros((n_bins, F), dtype=np.float32)

            # If multiple events hit same (row,col), the later in the list wins.
            # To make "last wins" deterministic by time, sort by (row) and stable index.
            order = np.argsort(rows, kind='mergesort')
            rows_sorted = rows[order]
            cols_sorted = cols[order]
            vals_sorted = vals[order]

            for r, c, v in zip(rows_sorted, cols_sorted, vals_sorted):
                temp_values[r, c] = v
                temp_masks[r, c]  = 1.0

            # Forward-fill across time; leading bins get start_fill=0.0
            values_ffill = _forward_fill(temp_values, start_fill=0.0)

            # Pack [values | masks]
            out = np.zeros((n_bins, 2 * F), dtype=np.float32)
            out[:, :F] = values_ffill
            out[:, F:] = temp_masks

            batch_out[i] = out

        return {
            'values': torch.from_numpy(batch_out),  # (B, n_bins, 2*F)
            'feature_names': self.feature_names
        }