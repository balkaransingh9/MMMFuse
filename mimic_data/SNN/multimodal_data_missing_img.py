import torch
from torch.utils.data import Dataset
import lmdb
from io import BytesIO
import pickle
import numpy as np

# fast JPEG decode + simple tensor transforms
import torchvision.io as tvio
import torchvision.transforms.v2 as T


class MultimodalData(Dataset):
    def __init__(
        self,
        list_file,
        modalities,               # e.g. ['physio','ecg','text'] or ['physio','medicine','text']
        task_type='phenotype',
        split='train',
        demographic_file = None,
        icd_code_file = None,
        split_col_name = 'original_split',
        lmdb_path_vital='none',
        lmdb_path_lab='none',
        lmdb_path_text='none',
        lmdb_path_medicine='none',
        lmdb_path_procedure='none',
        lmdb_path_output='none',
        lmdb_path_ecg='none',
        lmdb_path_cxr='none',
        ecg_normaliser=None,
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
            extra = ['subject_id','stay','period_length','stay_id','original_split','ecg_path', 
                     'ecg', 'text', 'med', 'output', 'procedure', 'lab', 'vital', 'cxr']
            self.sample_labels = torch.tensor(self.data_split.drop(extra,axis=1).values).float()
        else:
            self.sample_labels = torch.tensor(self.data_split['y_true'].values).float()

        # demographics/icd (optional)
        self.demographic_file = None
        if demographic_file is not None:
            self.demographic_file = demographic_file.set_index("stay_id")
            if 'subject_id' in self.demographic_file.columns:
                self.demographic_file = self.demographic_file.drop('subject_id', axis=1)

        self.icd_code_file = None
        if icd_code_file is not None:
            self.icd_code_file = icd_code_file.set_index("stay_id")

        # keys for LMDB
        self.vital_keys = [s.encode('utf-8') for s in self.data_split['vital'].astype(str)]
        self.lab_keys = [s.encode('utf-8') for s in self.data_split['lab'].astype(str)]
        self.procedure_keys = [s.encode('utf-8') for s in self.data_split['procedure'].astype(str)]
        self.output_keys = [s.encode('utf-8') for s in self.data_split['output'].astype(str)]
        self.med_keys = [s.encode('utf-8') for s in self.data_split['med'].astype(str)]
        self.text_keys = [s.encode('utf-8') for s in self.data_split['text'].astype(str)]
        self.ecg_keys = [s.encode('utf-8') for s in self.data_split['ecg'].astype(str)]
        self.cxr_keys = [s.encode('utf-8') for s in self.data_split['cxr'].astype(str)]

        if 'ecg' in modalities and ecg_normaliser is not None:
            self.m_ecg, self.s_ecg = ecg_normaliser['mean'], ecg_normaliser['std']

        # LMDB paths + env stubs
        self.lmdb_paths = {
            'vital':     lmdb_path_vital,
            'lab':       lmdb_path_lab,
            'procedure': lmdb_path_procedure,
            'output':    lmdb_path_output,
            'text':      lmdb_path_text,
            'medicine':  lmdb_path_medicine,
            'ecg':       lmdb_path_ecg,
            'cxr':       lmdb_path_cxr
        }
        self.envs = {mod: None for mod in self.lmdb_paths.keys()}
        self.txns = {}

        # transforms
        self._cxr_tf = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5], std=[0.25]),
        ])

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
            # one long-lived read txn per env (fast & safe for read-only)
            self.txns[mod] = self.envs[mod].begin(write=False, buffers=True)

    def _get(self, mod, key):
        self._open_env(mod)
        return self.txns[mod].get(key)

    # ---------- CXR: only decode the latest image ----------
    
    def _decode_jpeg_gray_fast(self, img_bytes: memoryview) -> torch.Tensor:
        """
        JPEG decode path:
        memoryview(bytes) -> np.frombuffer(...).copy() [writable] -> torch.from_numpy -> tvio.decode_jpeg
        Returns float32 normalized tensor AFTER transform: [1, 224, 224]
        """
        # Make a writable view (small copy of the *compressed* bytes, not the decoded image)
        arr = np.frombuffer(img_bytes, dtype=np.uint8).copy()   # <-- key change: .copy()
        buf = torch.from_numpy(arr)                             # uint8, CPU
        img_u8 = tvio.decode_jpeg(buf, mode=tvio.ImageReadMode.GRAY)  # [1,H,W] uint8
        img = self._cxr_tf(img_u8)                              # [1,224,224] float32 normalized
        return img

    def _load_cxr(self, idx):
        raw = self._get('cxr', self.cxr_keys[idx])
        if raw is None:
            return None, True

        # obj is expected to be a list of dicts: {'img': bytes, 'hours': float or None}
        obj = pickle.loads(raw)
        if not obj:
            return None, True

        # Prefer entry with max hours; if hours are missing, fallback to last element
        best_item = None
        max_hr = None
        any_hr_present = False
        for item in obj:
            hr = item.get('hours', None)
            if hr is not None:
                any_hr_present = True
                if (max_hr is None) or (hr > max_hr):
                    max_hr = hr
                    best_item = item

        if best_item is None:
            # No hours present; take last item
            best_item = obj[-1]
            hr_val = best_item.get('hours', None)
        else:
            hr_val = max_hr

        img_bytes = best_item['img']
        # Use memoryview to avoid copying
        img = self._decode_jpeg_gray_fast(memoryview(img_bytes))  # [1,224,224]
        hr_tensor = torch.tensor(0.0 if hr_val is None else float(hr_val), dtype=torch.float32)

        # Return a single image (+ hour), not a sequence
        return {"img": img, "hr": hr_tensor}, False

    # ---------- Other modalities (unchanged logic) ----------

    def _load_ecg(self, idx):
        raw = self._get('ecg', self.ecg_keys[idx])
        if raw is None:
            return None, True
        buffer = BytesIO(raw)
        loaded_data = torch.load(buffer)
        if hasattr(self, 'm_ecg') and hasattr(self, 's_ecg'):
            return (loaded_data['sample'] - self.m_ecg) / self.s_ecg, False
        return loaded_data['sample'], False
    
    def _load_demographic(self, idx):
        if self.demographic_file is None:
            return None, True
        stay_id = self.data_split.iloc[idx]['stay_id']
        if stay_id not in self.demographic_file.index:
            return None, True
        raw = self.demographic_file.loc[stay_id]
        if hasattr(raw, "isna") and raw.isna().all():
            return None, True
        return torch.tensor(raw.values, dtype=torch.float32), False

    def _load_icd_code(self, idx):
        if self.icd_code_file is None:
            return None, True
        stay_id = self.data_split.iloc[idx]['stay_id']
        if stay_id not in self.icd_code_file.index:
            return torch.zeros(self.icd_code_file.shape[1]), True
        raw = self.icd_code_file.loc[stay_id]
        if hasattr(raw, "isna") and raw.isna().all():
            return torch.zeros(self.icd_code_file.shape[1]), True
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

    def _load_procedure(self, idx):
        raw = self._get('procedure', self.procedure_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

    def _load_output(self, idx):
        raw = self._get('output', self.output_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

    def _load_medicine(self, idx):
        raw = self._get('medicine', self.med_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

    def _load_text(self, idx):
        raw = self._get('text', self.text_keys[idx])
        if raw is None:
            return "", True
        return pickle.loads(raw), False