import torch
from torch.utils.data import Dataset
import lmdb
from io import BytesIO

class PhysioData(Dataset):
  def __init__(self, list_file, task_type='phenotype', split = 'train', 
               normaliser_physio = None, lmdb_path_physio='none'):
    self.list_file = list_file
    self.split = split

    if split == 'train':
      self.data_split = self.list_file[self.list_file['original_split'] == 'train'].reset_index(drop=True)
    if split == 'test':
      self.data_split = self.list_file[self.list_file['original_split'] == 'test'].reset_index(drop=True)

    if task_type == 'phenotype':
      extra_cols = ['subject_id', 'stay', 'period_length', 'stay_id', 'original_split', 'ecg_path']
      self.sample_labels = torch.tensor(self.data_split.drop(extra_cols, axis=1).values).float()
    elif task_type == 'in_hospital_mortality':
      self.sample_labels = torch.tensor(self.data_split['y_true'].values).float()
    elif task_type == 'length_of_stay':
      self.sample_labels = torch.tensor(self.data_split['y_true'].values).float()
    else:
      raise ValueError("Unsupported task type!")      

    self.sample_keys = [i.split(".")[0].encode('utf-8') for i in self.data_split['stay'].values]
    self.sample_keys_text = [str(i).encode('utf-8') for i in self.data_split['stay_id'].values]

    self.m_physio = normaliser_physio['mean']
    self.stds_physio = normaliser_physio['std']

    self.lmdb_path_physio = lmdb_path_physio
    self.env_lmdb_data_physio = None

  def _normaliser_physio(self, input):
    norm = (input - self.m_physio)/self.stds_physio
    return norm

  def __len__(self):
    return len(self.data_split)

  def __getitem__(self, idx):
    #loading physio data
    if self.env_lmdb_data_physio == None:
      self.env_lmdb_data_physio = lmdb.open(self.lmdb_path_physio, readonly=True, lock=False)
      self.txn_data_physio = self.env_lmdb_data_physio.begin(write=False)

    retrieved_bytes = self.txn_data_physio.get(self.sample_keys[idx])
    if retrieved_bytes is not None:
      buffer = BytesIO(retrieved_bytes)
      loaded_data = torch.load(buffer, weights_only=True)
      normalised_sample_physio = self._normaliser_physio(loaded_data['sample'])

    return normalised_sample_physio, self.sample_labels[idx]