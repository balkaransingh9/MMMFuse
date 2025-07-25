import torch
from torch.utils.data import Dataset
import lmdb
from io import BytesIO
import pickle

class MultimodalData(Dataset):
    def __init__(
        self,
        list_file,
        modalities,               # e.g. ['physio','ecg','text'] or ['physio','medicine','text']
        task_type='phenotype',
        split='train',
        split_col_name = 'original_split',
        normaliser_physio=None,
        normaliser_ecg=None,
        lmdb_path_physio='none',
        lmdb_path_ecg='none',
        lmdb_path_text='none',
        lmdb_path_medicine='none',

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
        if task_type == 'phenotype':
            extra = ['subject_id','stay','period_length','stay_id','original_split','ecg_path', 'ecg', 'text', 'med']
            self.sample_labels = torch.tensor(self.data_split.drop(extra,axis=1).values).float()
        else:
            self.sample_labels = torch.tensor(self.data_split['y_true'].values).float()

        # keys for LMDB
        self.physio_keys = [s.split('.')[0].encode('utf-8') for s in self.data_split['stay'].astype(str)]
        self.med_keys = [s.encode('utf-8') for s in self.data_split['med'].astype(str)]
        self.text_keys = [s.encode('utf-8') for s in self.data_split['text'].astype(str)]
        self.ecg_keys =  [s.encode('utf-8') for s in self.data_split['ecg'].astype(str)]

        # normaliser params
        if 'physio' in modalities:
            self.m_physio, self.s_physio = normaliser_physio['mean'], normaliser_physio['std']
        if 'ecg'   in modalities:
            self.m_ecg,   self.s_ecg   = normaliser_ecg['mean'],   normaliser_ecg['std']

        # LMDB paths + env stubs
        self.lmdb_paths = {
            'physio':   lmdb_path_physio,
            'ecg':      lmdb_path_ecg,
            'text':     lmdb_path_text,
            'medicine': lmdb_path_medicine,
        }
        self.envs = {mod: None for mod in modalities}

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

    def _open_env(self, mod):
        # helper to lazily open each env
        if self.envs[mod] is None:
            self.envs[mod] = lmdb.open(self.lmdb_paths[mod], readonly=True, lock=False)
            self.envs[mod + "_txn"] = self.envs[mod].begin(write=False)

    def _load_physio(self, idx):
        self._open_env('physio')
        raw = self.envs['physio_txn'].get(self.physio_keys[idx])
        if raw is None:
            return None, True
        buf = BytesIO(raw)
        data = torch.load(buf, weights_only=True)['sample']
        return (data - self.m_physio) / self.s_physio, False

    def _load_ecg(self, idx):
        self._open_env('ecg')
        raw = self.envs['ecg_txn'].get(self.ecg_keys[idx])
        if raw is None:
            return None, True
        buf = BytesIO(raw)
        data = torch.load(buf, weights_only=True)['sample']
        return (data - self.m_ecg) / self.s_ecg, False

    def _load_medicine(self, idx):
        self._open_env('medicine')
        raw = self.envs['medicine_txn'].get(self.med_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

    def _load_text(self, idx):
        self._open_env('text')
        raw = self.envs['text_txn'].get(self.text_keys[idx])
        if raw is None:
            return "", True
        return raw.decode('utf-8'), False