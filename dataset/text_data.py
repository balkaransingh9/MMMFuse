import torch
from torch.utils.data import Dataset
import lmdb

class TextData(Dataset):
  def __init__(self, list_file, task_type = 'phenotype', split = 'train', lmdb_path_text='none'):
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

    self.lmdb_path_text = lmdb_path_text
    self.env_lmdb_data_text = None

  def __len__(self):
    return len(self.data_split)

  def __getitem__(self, idx):
    #loading text data
    if self.env_lmdb_data_text == None:
      self.env_lmdb_data_text = lmdb.open(self.lmdb_path_text, readonly=True, lock=False)
      self.txn_data_text = self.env_lmdb_data_text.begin(write=False)

    text_bytes = self.txn_data_text.get(self.sample_keys_text[idx])
    if text_bytes is not None:
      text = text_bytes.decode('utf-8')

    return text, self.sample_labels[idx]