import numpy as np
import torch

def _forward_fill_binary(values_2d: np.ndarray) -> np.ndarray:
    """
    Forward-fill a binary matrix per column (once 1, stays 1).
    values_2d: (R, C) of {0,1}
    """
    # cumulative OR along time dimension
    return np.minimum(1, np.cumsum(values_2d, axis=0))


class ProcedureTokenizer:
    """
    Tabular/discretized procedures grid with presence only.

    Per sample:
      - Time is binned into `duration`-hour bins over [0, tt_max].
      - For each (bin, procedure-label), mark presence=1 if any event falls in that bin.
      - No numeric value channel; we return [presence | mask] so it matches other tokenizers.
        (mask == presence here, i.e., observed-in-bin)

    Expected input per sample dict `m`:
      m['label']             : list[str]
      m['hours_from_intime'] : list[float]  (0..tt_max)

    Args:
      label_vocab : dict[str,int] defining columns (labels -> indices)

    tokenize(proc, tt_max=48, duration=1, ff_presence=False) -> {
      'values': FloatTensor(B, n_bins, 2*F)  where
                 [:, :, :F] = presence (0/1),
                 [:, :, F:] = mask (same as presence)
    }
    """
    def __init__(self, label_vocab, procnorm=None):
        self.label_vocab = label_vocab or {}

        # Stable column order: prefer index order if numeric, else alphabetical.
        try:
            items = sorted(self.label_vocab.items(), key=lambda kv: int(kv[1]))
        except Exception:
            items = sorted(self.label_vocab.items(), key=lambda kv: kv[0])

        self.labels_ordered = [k for k, _ in items]
        self.label2col = {lbl: i for i, lbl in enumerate(self.labels_ordered)}
        self.n_features = len(self.labels_ordered)

    def tokenize(self, proc, tt_max: int = 48, duration: int = 1, ff_presence: bool = False):
        """
        Args:
          proc       : list[dict|None]
          tt_max     : max hour horizon (inclusive end clamped into last bin)
          duration   : bin width in hours
          ff_presence: if True, forward-fills presence so once seen, it stays 1 thereafter.

        Returns:
          {'values': torch.FloatTensor(B, n_bins, 2*F)}
        """
        B = len(proc)
        F = self.n_features
        n_bins = int(tt_max // duration)

        # (B, n_bins, 2*F): [presence | mask]
        # batch_out = np.zeros((B, n_bins, 2 * F), dtype=np.float32)
        batch_out = np.zeros((B, n_bins, F), dtype=np.float32)

        for i, m in enumerate(proc):
            if not m or F == 0:
                continue

            labels = m.get('label', [])
            hours  = m.get('hours_from_intime', [])
            if len(labels) == 0:
                continue

            labels = np.asarray(labels, dtype=object)
            hours  = np.asarray(hours,  dtype=np.float32)

            # Keep only known labels
            keep = np.array([l in self.label2col for l in labels], dtype=bool)
            if not keep.any():
                continue

            labels = labels[keep]
            hours  = hours[keep]
            cols   = np.array([self.label2col[l] for l in labels], dtype=np.int64)

            # Bin rows; clamp tt == tt_max into last bin
            rows = np.floor(hours / duration).astype(np.int64)
            rows = np.clip(rows, 0, n_bins - 1)

            # Presence matrix per sample
            presence = np.zeros((n_bins, F), dtype=np.float32)
            # Mark 1 for any (row, col) that has an event
            # If duplicates, still just 1.
            presence[rows, cols] = 1.0

            if ff_presence:
                presence = _forward_fill_binary(presence)

            # Pack [presence | mask], with mask == presence (observed-in-bin)
            # out_mat = np.zeros((n_bins, 2 * F), dtype=np.float32)
            # out_mat[:, :F] = presence
            # out_mat[:, F:] = presence  # same as mask
            out_mat = presence.astype(np.float32)

            batch_out[i] = out_mat

        return {
            'values': torch.from_numpy(batch_out)  # (B, n_bins, 2*F)
        }
