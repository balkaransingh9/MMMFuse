from io import BytesIO
import torch
from tqdm import tqdm

def welford_update(curr_sample, n_old, m_old, s_old):
  n_new = n_old + len(curr_sample)
  m_new = m_old + (curr_sample - m_old).sum(axis=0)/n_new
  s_new = s_old + ((curr_sample - m_new) * (curr_sample - m_old)).sum(axis=0)
  return n_new, m_new, s_new

def normaliser(train_samples, lmdb_env, ftype_data):  
    n, m, s = 0, 0, 0
    for sample in tqdm(train_samples):
        with lmdb_env.begin(write=False) as txn:
            retrieved_bytes = txn.get(sample)
            if retrieved_bytes is not None:               
                buffer = BytesIO(retrieved_bytes)
                loaded_data = torch.load(buffer)
                n, m, s = welford_update(loaded_data['sample'], n, m, s)
        stds = torch.sqrt(s/(n-1))

    if ftype_data is not None:
        cont_mask = torch.tensor([1 if ftype_data[i]=='continuous' else 0 for i in ftype_data['column_names']])
        mean = cont_mask*m
        std = stds.clone()
        std[cont_mask==0] = 1

        norm_parms = {'mean': mean, 'std': std}
    else:
        norm_parms = {'mean': m, 'std': stds}
    
    return norm_parms