import torch
from torch.utils.data import Dataset
import lmdb
from io import BytesIO

class MultimodalData(Dataset):
  def __init__(self, list_file, task_type = 'phenotype', split = 'train', 
               normaliser_physio = None, normaliser_ecg = None,
               lmdb_path_physio='none', lmdb_path_ecg='none', lmdb_path_text='none'):
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

    self.m_ecg = normaliser_ecg['mean']
    self.stds_ecg = normaliser_ecg['std']

    self.lmdb_path_physio = lmdb_path_physio
    self.lmdb_path_ecg = lmdb_path_ecg
    self.lmdb_path_text = lmdb_path_text

    self.env_lmdb_data_physio = None
    self.env_lmdb_data_ecg = None
    self.env_lmdb_data_text = None

  def _normaliser_physio(self, input):
    norm = (input - self.m_physio)/self.stds_physio
    return norm

  def _normaliser_ecg(self, input):
    norm = (input - self.m_ecg)/self.stds_ecg
    return norm

  def __len__(self):
    return len(self.data_split)

  def __getitem__(self, idx):
    #--loading physio data--
    physio_missing_flag = False
    if self.env_lmdb_data_physio == None:
      self.env_lmdb_data_physio = lmdb.open(self.lmdb_path_physio, readonly=True, lock=False)
      self.txn_data_physio = self.env_lmdb_data_physio.begin(write=False)

    retrieved_bytes = self.txn_data_physio.get(self.sample_keys[idx])
    if retrieved_bytes is not None:
      buffer = BytesIO(retrieved_bytes)
      loaded_data = torch.load(buffer, weights_only=True)
      normalised_sample_physio = self._normaliser_physio(loaded_data['sample'])
    else:
      normalised_sample_physio = None
      physio_missing_flag = True

    #--loading ecg data--
    ecg_missing_flag = False
    if self.env_lmdb_data_ecg == None:
      self.env_lmdb_data_ecg = lmdb.open(self.lmdb_path_ecg, readonly=True, lock=False)
      self.txn_data_ecg = self.env_lmdb_data_ecg.begin(write=False)

    retrieved_bytes = self.txn_data_ecg.get(self.sample_keys[idx])
    if retrieved_bytes is not None:
      buffer = BytesIO(retrieved_bytes)
      loaded_data = torch.load(buffer, weights_only=True)
      normalised_sample_ecg = self._normaliser_ecg(loaded_data)
    else:
      normalised_sample_ecg = None
      ecg_missing_flag = True

    #--loading text data--
    text_missing_flag = False
    if self.env_lmdb_data_text == None:
      self.env_lmdb_data_text = lmdb.open(self.lmdb_path_text, readonly=True, lock=False)
      self.txn_data_text = self.env_lmdb_data_text.begin(write=False)

    text_bytes = self.txn_data_text.get(self.sample_keys_text[idx])
    if text_bytes is not None:
      text = text_bytes.decode('utf-8')
    else:
      text = ""
      text_missing_flag = True

    missing_flag_dict = {'physio_missing': physio_missing_flag, 'ecg_missing': ecg_missing_flag, 'text_missing': text_missing_flag}
    return normalised_sample_physio, normalised_sample_ecg, text, missing_flag_dict, self.sample_labels[idx]