import torch
from torch.utils.data import Dataset
import lmdb
import pickle

class MultimodalData(Dataset):
    def __init__(
        self,
        list_file,
        modalities,               # e.g. ['vital','lab','medicine','demographic','diagnosis','treatment']
        task_type='mortality',
        split='train',
        split_col_name='original_split',
        lmdb_path_vital='none',
        lmdb_path_lab='none',
        lmdb_path_medicine='none',
        demographic_file=None,
        diagnosis_file=None,
        treatment_file=None,
        vital_norm_stats=None,
        lab_norm_stats=None,
    ):
        self.list_file      = list_file
        self.modalities     = modalities
        self.task_type      = task_type
        self.split          = split
        self.split_col_name = split_col_name

        # split the DataFrame
        if self.split == 'train':
            self.data_split = list_file[list_file[self.split_col_name] == 'train'].reset_index(drop=True)
        elif self.split == 'val':
            self.data_split = list_file[list_file[self.split_col_name] == 'val'].reset_index(drop=True)
        else:
            self.data_split = list_file[list_file[self.split_col_name] == 'test'].reset_index(drop=True)

        # labels
        if task_type == 'mortality':
            self.sample_labels = torch.tensor(self.data_split['mortality'].values).float()
        elif task_type == 'readmission':
            self.sample_labels = torch.tensor(self.data_split['readmission'].values).float()
        else:
            raise ValueError("Task not supported")

        # static tables -> index by stay id for O(1) lookup
        self.demographic_file = None
        if demographic_file is not None:
            self.demographic_file = demographic_file.set_index("patientunitstayid")

        self.diagnosis_file = None
        if diagnosis_file is not None:
            self.diagnosis_file = diagnosis_file.set_index("patientunitstayid")

        self.treatment_file = None
        if treatment_file is not None:
            self.treatment_file = treatment_file.set_index("patientunitstayid")

        # keys for LMDB (store as bytes)
        self.vital_keys    = [s.encode('utf-8') for s in self.data_split['vital'].astype(str)]
        self.lab_keys      = [s.encode('utf-8') for s in self.data_split['lab'].astype(str)]
        self.medicine_keys = [s.encode('utf-8') for s in self.data_split['medicine'].astype(str)]

        # LMDB paths + env stubs
        self.lmdb_paths = {
            'vital':    lmdb_path_vital,
            'lab':      lmdb_path_lab,
            'medicine': lmdb_path_medicine,
        }
        self.envs = {mod: None for mod in self.lmdb_paths.keys()}
        self.txns = {}

        # normalization stats (dicts with tensors)
        self.vital_norm_stats = vital_norm_stats
        self.lab_norm_stats   = lab_norm_stats

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        out = {}
        flags = {}

        for mod in self.modalities:
            loader = getattr(self, f"_load_{mod}")
            data, missed = loader(idx)
            out[mod] = data
            flags[f"{mod}_missing"] = missed

        label = self.sample_labels[idx]
        return out, flags, label

    # ---------- LMDB helpers ----------

    def _open_env(self, mod):
        # Lazily open each env with perf-friendly flags
        if self.envs.get(mod) is None:
            self.envs[mod] = lmdb.open(
                self.lmdb_paths[mod],
                readonly=True,
                lock=False,
                readahead=False,     # better for random reads
                max_readers=2048,
                map_async=True,
            )
            self.txns[mod] = self.envs[mod].begin(write=False, buffers=True)

    def _get(self, mod, key):
        self._open_env(mod)
        return self.txns[mod].get(key)

    # Normalization helpers 

    @staticmethod
    def _standardize_value_tensor(val: torch.Tensor, stats: dict, impute: bool = True):
        """
        val: (..., D) float tensor with possible NaNs/inf. D must match stats['mean'].shape[0].
        If impute=True:
          - replace NaNs with per-channel mean BEFORE z-score
          - compute z-score
          - zero-out positions that were originally missing
        Returns: norm_val (float32), mask (float32) where mask==1 if original value was finite
        """
        val = val.to(torch.float32)

        mean = stats['mean'].to(dtype=torch.float32, device=val.device)
        std  = stats['std'].to(dtype=torch.float32, device=val.device).clamp_min(1e-6)

        if val.shape[-1] != mean.shape[0]:
            raise ValueError(f"Feature dim mismatch: val has {val.shape[-1]}, stats has {mean.shape[0]}")

        mask = torch.isfinite(val)  # True where value is not nan/inf
        if impute:
            val_filled = torch.where(mask, val, mean)  # broadcast across feature dim
            norm = (val_filled - mean) / std
            norm = torch.where(mask, norm, torch.zeros_like(norm))
        else:
            norm = val.clone()
            norm[mask] = (val[mask] - mean) / std
            # keep NaNs as-is if not imputing

        return norm, mask.to(torch.float32)

    def _normalize_payload_if_any(self, payload, stats):
        """
        payload is often a dict with 'value' (and maybe 'time' or other fields), or a raw array/tensor.
        We normalize the last dimension using stats; attach a 'mask' indicating originally present values.
        Returns:
          - dict with at least 'value' and 'mask' if normalization applied,
          - original payload if stats is None,
          - None if payload is None.
        """
        if payload is None or stats is None:
            return payload

        if isinstance(payload, dict) and 'value' in payload:
            val = torch.as_tensor(payload['value'], dtype=torch.float32)
            norm, mask = self._standardize_value_tensor(val, stats, impute=True)
            payload = dict(payload)  # shallow copy to avoid mutating cached objects
            payload['value'] = norm
            payload['mask']  = mask
            return payload

        val = torch.as_tensor(payload, dtype=torch.float32)
        norm, mask = self._standardize_value_tensor(val, stats, impute=True)
        return {'value': norm, 'mask': mask}

    # Static modalities

    def _load_demographic(self, idx):
        if self.demographic_file is None:
            return None, True
        stay_id = self.data_split.iloc[idx]['patientunitstayid']
        if stay_id not in self.demographic_file.index:
            return None, True
        raw = self.demographic_file.loc[stay_id]
        if hasattr(raw, "isna") and raw.isna().all():
            return None, True
        return torch.tensor(raw.values, dtype=torch.float32), False

    def _load_diagnosis(self, idx):
        if self.diagnosis_file is None:
            return None, True
        stay_id = self.data_split.iloc[idx]['patientunitstayid']
        if stay_id not in self.diagnosis_file.index:
            return torch.zeros(self.diagnosis_file.shape[1]), True
        raw = self.diagnosis_file.loc[stay_id]
        if hasattr(raw, "isna") and raw.isna().all():
            return torch.zeros(self.diagnosis_file.shape[1]), True
        return torch.tensor(raw.values, dtype=int), False

    def _load_treatment(self, idx):
        if self.treatment_file is None:
            return None, True
        stay_id = self.data_split.iloc[idx]['patientunitstayid']
        if stay_id not in self.treatment_file.index:
            return torch.zeros(self.treatment_file.shape[1]), True
        raw = self.treatment_file.loc[stay_id]
        if hasattr(raw, "isna") and raw.isna().all():
            return torch.zeros(self.treatment_file.shape[1]), True
        return torch.tensor(raw.values, dtype=int), False

    # Time-series modalities 

    def _load_vital(self, idx):
        raw = self._get('vital', self.vital_keys[idx])
        if raw is None:
            return None, True
        payload = pickle.loads(raw)
        payload = self._normalize_payload_if_any(payload, self.vital_norm_stats)
        return payload, False

    def _load_lab(self, idx):
        raw = self._get('lab', self.lab_keys[idx])
        if raw is None:
            return None, True
        payload = pickle.loads(raw)
        payload = self._normalize_payload_if_any(payload, self.lab_norm_stats)
        return payload, False

    def _load_medicine(self, idx):
        raw = self._get('medicine', self.medicine_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False