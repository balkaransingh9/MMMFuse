from io import BytesIO
import lmdb
import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

class LmdbSampleDataset(IterableDataset):
    def __init__(self, lmdb_env: lmdb.Environment, keys: list[bytes]):
        self.env  = lmdb_env
        self.keys = keys

    def __iter__(self):
        txn = self.env.begin(write=False)
        for key in self.keys:
            data = txn.get(key)
            if data is not None:
                sample = torch.load(BytesIO(data))['sample']
                yield sample  # shape: (B_i, D)

def normaliser(
    train_samples: list[bytes],
    lmdb_env: lmdb.Environment,
    ftype_data: dict | None = None,
    num_workers: int = 4
) -> dict[str, torch.Tensor]:
    """
    Compute per‐feature mean & sample‐std over all train_samples in lmdb_env.
    If ftype_data is provided, it must have:
      - 'column_names': list of str
      - mapping from each column_name to its type ('continuous' or else)
    Non‐continuous features get mean=0, std=1.
    """
    # 1) Set up parallel loader (no batching — yields each tensor as is)
    dataset = LmdbSampleDataset(lmdb_env, train_samples)
    loader  = DataLoader(dataset,
                         num_workers=num_workers,
                         batch_size=None)

    # 2) Accumulators
    n = 0
    sum_x  = None
    sum_x2 = None

    # 3) Loop with progress bar
    for sample in tqdm(loader, total=len(train_samples)):
        B, D = sample.shape

        if sum_x is None:
            sum_x  = torch.zeros(D, dtype=sample.dtype)
            sum_x2 = torch.zeros(D, dtype=sample.dtype)

        n      += B
        sum_x  += sample.sum(dim=0)
        sum_x2 += (sample * sample).sum(dim=0)

    if n < 2:
        raise ValueError("Need at least two total rows to compute sample‐std.")

    mean = sum_x / n
    var_sample = (sum_x2 - sum_x * sum_x / n) / (n - 1)
    std  = torch.sqrt(var_sample)

    if ftype_data is not None:
        cont_mask = torch.tensor(
            [1 if ftype_data[col] == 'continuous' else 0
             for col in ftype_data['column_names']],
            dtype=torch.bool
        )
        # zero‐out means for non‐continuous, set std=1 there
        mean = mean * cont_mask.to(mean.dtype)
        std  = std.clone()
        std[~cont_mask] = 1.0

    return {'mean': mean, 'std': std}