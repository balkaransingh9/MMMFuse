import torch
from torch.utils.data import Dataset
import lmdb
import pickle

class MultimodalData(Dataset):
    def __init__(
        self,
        list_file,
        modalities,               # e.g. ['physio','ecg','text'] or ['physio','medicine','text']
        task_type='mortality',
        split='train',
        split_col_name = 'original_split',
        lmdb_path_vital='none',
        lmdb_path_lab='none',
        lmdb_path_medicine='none',
        demographic_file = None,
        diagnosis_file = None,
        treatment_file = None
    ):
        self.list_file      = list_file
        self.modalities     = modalities
        self.task_type      = task_type
        self.split          = split  # assume `list_file` has an 'original_split' column
        self.split_col_name = split_col_name

        # split the DataFrame
        if self.split == 'train':
            self.data_split = list_file[list_file[self.split_col_name]=='train'].reset_index(drop=True)
        elif self.split == 'val':
            self.data_split = list_file[list_file[self.split_col_name]=='val'].reset_index(drop=True)
        else:
            self.data_split = list_file[list_file[self.split_col_name]=='test'].reset_index(drop=True)

        # labels
        if task_type == 'mortality':
            self.sample_labels = torch.tensor(self.data_split['mortality'].values).float()
        elif task_type == 'readmission':
            self.sample_labels = torch.tensor(self.data_split['readmission'].values).float()
        else:
            raise ValueError("Task not supported")

        self.demographic_file = None
        if demographic_file is not None:
            self.demographic_file = demographic_file.set_index("patientunitstayid")

        self.diagnosis_file = None
        if diagnosis_file is not None:
            self.diagnosis_file = diagnosis_file.set_index("patientunitstayid")

        self.treatment_file = None
        if treatment_file is not None:
            self.treatment_file = treatment_file.set_index("patientunitstayid")

        # keys for LMDB
        self.vital_keys = [s.encode('utf-8') for s in self.data_split['vital'].astype(str)]
        self.lab_keys = [s.encode('utf-8') for s in self.data_split['lab'].astype(str)]
        self.medicine_keys = [s.encode('utf-8') for s in self.data_split['medicine'].astype(str)]

        # LMDB paths + env stubs
        self.lmdb_paths = {
            'vital':     lmdb_path_vital,
            'lab':       lmdb_path_lab,
            'medicine':  lmdb_path_medicine,
        }
        self.envs = {mod: None for mod in self.lmdb_paths.keys()}
        self.txns = {}

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        out = {}
        flags = {}

        for mod in self.modalities:
            loader = getattr(self, f"_load_{mod}")
            data, missed = loader(idx)
            out[mod]       = data
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
                readahead=False,     # good for random reads
                max_readers=2048,
                map_async=True,
            )
            self.txns[mod] = self.envs[mod].begin(write=False, buffers=True)

    def _get(self, mod, key):
        self._open_env(mod)
        return self.txns[mod].get(key)
    
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

    def _load_vital(self, idx):
        raw = self._get('vital', self.vital_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

    def _load_lab(self, idx):
        raw = self._get('lab', self.lab_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

    def _load_medicine(self, idx):
        raw = self._get('medicine', self.medicine_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False