import torch
from torch.utils.data import Dataset
import lmdb
from io import BytesIO
import pickle

# Added imports for fast JPEG decode + simple tensor transforms
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
                     'ecg', 'text', 'med', 'output', 'procedure', 'lab', 'vital']
            
            self.sample_labels = torch.tensor(self.data_split.drop(extra,axis=1).values).float()
        else:
            self.sample_labels = torch.tensor(self.data_split['y_true'].values).float()

        self.demographic_file = demographic_file.set_index("stay_id")
        self.demographic_file = self.demographic_file.drop('subject_id', axis=1)

        if icd_code_file is not None:
            self.icd_code_file = icd_code_file
            self.icd_code_file = self.icd_code_file.set_index("stay_id")

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
            'vital':   lmdb_path_vital,
            'lab':      lmdb_path_lab,
            'procedure': lmdb_path_procedure,
            'output':    lmdb_path_output,
            'text':     lmdb_path_text,
            'medicine': lmdb_path_medicine,
            'ecg':      lmdb_path_ecg,
            'img':      lmdb_path_cxr
        }
        self.envs = {mod: None for mod in modalities}

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

    def _open_env(self, mod):
        # helper to lazily open each env
        if self.envs[mod] is None:
            self.envs[mod] = lmdb.open(self.lmdb_paths[mod], readonly=True, lock=False)
            self.envs[mod + "_txn"] = self.envs[mod].begin(write=False)

    def _load_cxr(self, idx):
        """
        Loads CXR images + hours from LMDB.
        Each LMDB value is expected to be a pickled list of dicts with keys.
        """
        self._open_env('img')
        raw = self.envs['img_txn'].get(self.img_keys[idx])
        if raw is None:
            return None, True

        obj = pickle.loads(raw)
        images, hours = [], []
        for item in obj:
            img_bytes = item['img']
            hr_val    = item['hours']
            img_u8 = tvio.decode_jpeg(
                torch.frombuffer(img_bytes, dtype=torch.uint8),
                mode=tvio.ImageReadMode.GRAY
            )
            img = self._cxr_tf(img_u8)  # [1,224,224]
            images.append(img)
            if hr_val is not None:
                hours.append(float(hr_val))

        if len(images) == 0:
            return None, True

        #sort if we have both imgs and hrs
        if len(images) > 1 and len(hours) == len(images):
            hrs_tensor = torch.tensor(hours, dtype=torch.float32)
            order = torch.argsort(hrs_tensor)  # ascending; use descending for latest-first
            images = [images[i] for i in order]
            hours  = hrs_tensor[order].tolist()

        images = torch.stack(images, dim=0)              # [N,1,224,224]
        hrs    = torch.tensor(hours, dtype=torch.float32) if hours else torch.empty(0)
        return {"img": images, "hrs": hrs}, False

    def _load_ecg(self, idx):
        self._open_env('ecg')
        raw = self.envs['ecg_txn'].get(self.ecg_keys[idx])
        if raw is None:
            return None, True
        buffer = BytesIO(raw)
        loaded_data = torch.load(buffer)
        return (loaded_data['sample'] - self.m_ecg) / self.s_ecg, False
    
    def _load_demographic(self, idx):
        stay_id = self.data_split.iloc[idx]['stay_id']
        if stay_id not in self.demographic_file.index:
            return None, True
        raw = self.demographic_file.loc[stay_id]
        if raw.isna().all():
            return None, True
        return torch.tensor(raw.values, dtype=torch.float32), False

    def _load_icd_code(self, idx):
        stay_id = self.data_split.iloc[idx]['stay_id']
        if stay_id not in self.icd_code_file.index:
            return torch.zeros(self.icd_code_file.shape[1]), True
        raw = self.icd_code_file.loc[stay_id]
        if raw.isna().all():
            return torch.zeros(self.icd_code_file.shape[1]), True
        return torch.tensor(raw.values, dtype=int), False

    def _load_vital(self, idx):
        self._open_env('vital')
        raw = self.envs['vital_txn'].get(self.vital_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

    def _load_lab(self, idx):
        self._open_env('lab')
        raw = self.envs['lab_txn'].get(self.lab_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

    def _load_procedure(self, idx):
        self._open_env('procedure')
        raw = self.envs['procedure_txn'].get(self.procedure_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

    def _load_output(self, idx):
        self._open_env('output')
        raw = self.envs['output_txn'].get(self.output_keys[idx])
        if raw is None:
            return None, True
        return pickle.loads(raw), False

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
        return pickle.loads(raw), False