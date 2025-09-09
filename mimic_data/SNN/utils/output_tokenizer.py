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


class OutputTokenizer:
    """
    Tabular/discretized outputs grid (same pattern as your meds/vitals reg_ts).

    Per sample:
      - Time is binned into duration-hour bins over [0, tt_max].
      - For each (bin, output-label), the last observed value wins.
      - Then we forward-fill across time (LOCF).
      - Output shape per sample: (n_bins, 2*F) = [values_ffill | masks].

    Required input keys per sample dict `m`:
      m['label']             : list[str]
      m['value']             : list[float]
      m['hours_from_intime'] : list[float]  (0..tt_max)

    Args:
      label_vocab : dict[str,int] (used to define the column order)
      outnorm     : dict with outnorm['value']['mean'], outnorm['value']['std'] (scalars)

    tokenize(out, tt_max=48, duration=1) returns:
      {'values': FloatTensor(B, n_bins, 2*F)}
    """
    def __init__(self, label_vocab, outnorm):
        self.label_vocab = label_vocab or {}
        self.outnorm = outnorm or {'value': {'mean': 0.0, 'std': 1.0}}

        # Stable column order: try index order if indices are numeric/contiguous, else by label name
        try:
            items = sorted(self.label_vocab.items(), key=lambda kv: int(kv[1]))
        except Exception:
            items = sorted(self.label_vocab.items(), key=lambda kv: kv[0])

        self.labels_ordered = [k for k, _ in items]
        self.label2col = {lbl: i for i, lbl in enumerate(self.labels_ordered)}
        self.n_features = len(self.labels_ordered)

        self._val_mean = float(self.outnorm.get('value', {}).get('mean', 0.0))
        self._val_std  = float(self.outnorm.get('value', {}).get('std',  1.0))

    def tokenize(self, out, tt_max: int = 48, duration: int = 1):
        """
        Args:
          out: list[dict|None]
          tt_max: max hour horizon (inclusive end clamped into last bin)
          duration: bin width in hours

        Returns:
          {'values': torch.FloatTensor(B, n_bins, 2*F)}
        """
        B = len(out)
        F = self.n_features
        n_bins = int(tt_max // duration)

        batch_out = np.zeros((B, n_bins, 2 * F), dtype=np.float32)

        for i, m in enumerate(out):
            if not m:
                continue

            labels = m.get('label', [])
            vals   = m.get('value', [])
            hours  = m.get('hours_from_intime', [])

            if len(labels) == 0:
                continue

            labels = np.asarray(labels, dtype=object)
            vals   = np.asarray(vals,   dtype=np.float32)
            hours  = np.asarray(hours,  dtype=np.float32)

            # normalize values (global scalar mean/std)
            vals = (vals - self._val_mean) / (self._val_std + 1e-8)

            # Map labels to columns; drop unknowns
            keep_mask = np.array([l in self.label2col for l in labels], dtype=bool)
            if not keep_mask.any():
                continue

            labels = labels[keep_mask]
            vals   = vals[keep_mask]
            hours  = hours[keep_mask]
            cols   = np.array([self.label2col[l] for l in labels], dtype=np.int64)

            # Bin rows; clamp tt == tt_max into last bin
            rows = np.floor(hours / duration).astype(np.int64)
            rows = np.clip(rows, 0, n_bins - 1)

            # last-observation-wins per (row, col)
            temp_values = np.full((n_bins, F), np.nan, dtype=np.float32)
            temp_masks  = np.zeros((n_bins, F), dtype=np.float32)

            # Sort by row (stable) to make overwrite order deterministic by time
            order = np.argsort(rows, kind='mergesort')
            rows_sorted = rows[order]
            cols_sorted = cols[order]
            vals_sorted = vals[order]

            for r, c, v in zip(rows_sorted, cols_sorted, vals_sorted):
                temp_values[r, c] = v
                temp_masks[r, c]  = 1.0

            # Forward-fill; leading bins get start_fill = 0.0
            values_ffill = _forward_fill(temp_values, start_fill=0.0)

            # out_mat = np.zeros((n_bins, 2 * F), dtype=np.float32)
            # out_mat[:, :F] = values_ffill
            # out_mat[:, F:] = temp_masks
            out_mat = values_ffill.astype(np.float32)

            batch_out[i] = out_mat

        return {
            'values': torch.from_numpy(batch_out)  # (B, n_bins, 2*F)
        }