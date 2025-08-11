import lmdb
import pickle
from tqdm.auto import tqdm
from collections import Counter

def build_vocab(lmdb_path, feature_key, min_freq=1):
  env = lmdb.open(lmdb_path, readonly=True)

  all_feature = []
  with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in tqdm(cursor):
      item = pickle.loads(value)
      item_feature = item[feature_key]
      all_feature.extend(item_feature)

  cnt = Counter(all_feature)
  toks = [t for t,c in cnt.items() if c>=min_freq]
  vocab = {'<PAD>':0, '<UNK>':1}
  for t in toks:
    vocab[t] = len(vocab)

  return vocab