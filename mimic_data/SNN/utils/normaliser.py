from io import BytesIO
import torch
from tqdm.auto import tqdm
import numpy as np
import lmdb
import pickle

def welford_update(curr_sample, n_old, m_old, s_old):
  n_new = n_old + len(curr_sample)
  m_new = m_old + (curr_sample - m_old).sum(axis=0)/n_new
  s_new = s_old + ((curr_sample - m_new) * (curr_sample - m_old)).sum(axis=0)
  return n_new, m_new, s_new

def normaliser(listfile, split_col='original_split', lmdb_path=None, ftype_data=None):

    listfile_train = listfile[listfile[split_col] == 'train']
    train_samples = [i.split(".")[0].encode('utf-8') for i in listfile_train['stay']]
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    n, m, s = 0, 0, 0
    for sample in tqdm(train_samples):
        with env.begin(write=False) as txn:
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


def med_normaliser(listfile_df, lmdb_path, split_feature='original_split'):
    train_df = listfile_df[listfile_df[split_feature] == 'train']
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn, txn.cursor() as cursor:
        lmdb_ids = {int(key.decode('utf-8')) for key, _ in cursor}

    train_df = train_df[train_df['stay_id'].isin(lmdb_ids)]
    train_keys = [str(sid).encode('utf-8') for sid in train_df['stay_id']]

    all_values = []
    all_hours = []

    with env.begin() as txn:
        for key in train_keys:
            raw = txn.get(key)
            if raw is None:
                continue
            item = pickle.loads(raw)
            all_values.extend(item.get('amount_std_value', []))
            all_hours.extend(item.get('hours_from_intime', []))

    env.close()

    if not all_values or not all_hours:
        raise RuntimeError("No medication records found for any training stay.")

    vals = np.array(all_values)
    hrs  = np.array(all_hours)

    return {
        'value': {
            'mean': vals.mean(),
            'std':   vals.std(ddof=0)
        },
        'hours': {
            'mean': hrs.mean(),
            'std':   hrs.std(ddof=0)
        }
    }

def vital_normaliser(listfile_df, lmdb_path, discrete_labels):
    """
    Scans all training stays in LMDB and computes:
      - per‐label mean/std for continuous vitals only
      - global mean/std of hours_from_intime
    """
    # 1) Filter to train stays in LMDB
    train_df = listfile_df[listfile_df['original_split']=='train']
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn, txn.cursor() as cur:
        lmdb_ids = {int(k.decode()): None for k,_ in cur}
    train_df = train_df[train_df['stay_id'].isin(lmdb_ids)]
    stay_keys = [str(s).encode() for s in train_df['stay_id']]
    
    # 2) Accumulate raw values
    per_label_vals = {}   # label_name → list of floats
    all_hours = []
    
    with env.begin() as txn:
        for key in stay_keys:
            raw = txn.get(key)
            if raw is None:
                continue
            item = pickle.loads(raw)
            
            labels = item.get('label', [])
            values = item.get('value', [])
            hours  = item.get('hours_from_intime', [])
            
            # record hours
            all_hours.extend(hours)
            
            # for each (label, value) pair
            for lbl, val in zip(labels, values):
                if lbl in discrete_labels:
                    continue  # skip strings entirely
                try:
                    fv = float(val)
                except Exception:
                    continue  # skip any non‑floatable junk
                per_label_vals.setdefault(lbl, []).append(fv)
    
    env.close()
    
    # 3) Compute global hours stats
    if not all_hours:
        raise RuntimeError("No hours found in training data.")
    hrs = np.array(all_hours, dtype=float)
    hours_mean = float(hrs.mean())
    hours_std  = float(hrs.std())
    
    # 4) Compute per‑label continuous stats
    value_stats = {}
    for lbl, arr in per_label_vals.items():
        if len(arr) < 2:
            # not enough data
            continue
        a = np.array(arr, dtype=float)
        value_stats[lbl] = {
            'mean': float(a.mean()),
            'std':  float(a.std())
        }
    
    return {
        'value': value_stats,        # e.g. {'Heart Rate': {'mean':..., 'std':...}, ...}
        'hours': {'mean': hours_mean, 'std': hours_std}
    }

def value_hour_normaliser(listfile_df, lmdb_path, split_feature='original_split'):
    train_df = listfile_df[listfile_df[split_feature] == 'train']
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn, txn.cursor() as cursor:
        lmdb_ids = {int(key.decode('utf-8')) for key, _ in cursor}

    train_df = train_df[train_df['stay_id'].isin(lmdb_ids)]
    train_keys = [str(sid).encode('utf-8') for sid in train_df['stay_id']]

    all_values = []
    all_hours = []

    with env.begin() as txn:
        for key in train_keys:
            raw = txn.get(key)
            if raw is None:
                continue
            item = pickle.loads(raw)
            all_values.extend(item.get('value', []))
            all_hours.extend(item.get('hours_from_intime', []))

    env.close()

    if not all_values or not all_hours:
        raise RuntimeError("No labs records found for any training stay.")

    vals = np.array(all_values)
    hrs  = np.array(all_hours)

    return {
        'value': {
            'mean': vals.mean(),
            'std':   vals.std(ddof=0)
        },
        'hours': {
            'mean': hrs.mean(),
            'std':   hrs.std(ddof=0)
        }
    }

def procedure_normaliser(listfile_df, lmdb_path, split_feature='original_split'):
    train_df = listfile_df[listfile_df[split_feature] == 'train']
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn, txn.cursor() as cursor:
        lmdb_ids = {int(key.decode('utf-8')) for key, _ in cursor}

    train_df = train_df[train_df['stay_id'].isin(lmdb_ids)]
    train_keys = [str(sid).encode('utf-8') for sid in train_df['stay_id']]

    all_duration = []
    all_hours = []

    with env.begin() as txn:
        for key in train_keys:
            raw = txn.get(key)
            if raw is None:
                continue
            item = pickle.loads(raw)
            all_duration.extend(item.get('procedure_duration', []))
            all_hours.extend(item.get('hours_from_intime', []))

    env.close()

    if not all_duration or not all_hours:
        raise RuntimeError("No labs records found for any training stay.")

    duration = np.array(all_duration)
    hrs  = np.array(all_hours)

    return {
        'duration': {
            'mean': duration.mean(),
            'std':   duration.std(ddof=0)
        },
        'hours': {
            'mean': hrs.mean(),
            'std':   hrs.std(ddof=0)
        }
    }