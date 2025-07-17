import torch

class VitalTokenizer:
    """
    Turns a list of raw-vitals dicts into padded tensors:
      - 'label'     : LongTensor (B, L)
      - 'value'     : FloatTensor (B, L)  (only continuous entries are meaningful)
      - 'value_id'  : LongTensor (B, L)   (only discrete entries are meaningful)
      - 'hours'     : FloatTensor (B, L)
      - 'mask'      : BoolTensor  (B, L)  True where padding
    """
    def __init__(self,
        label_vocab: dict[str,int],
        vitalnorm: dict,
        discrete_label_categorical_values: dict[str,list[str]],
        pad_idx: int = 0
    ):
        self.label_vocab = label_vocab
        self.pad_idx     = pad_idx

        # normalization stats
        self.val_stats   = vitalnorm['value']
        self.hrs_mean    = vitalnorm['hours']['mean']
        self.hrs_std     = vitalnorm['hours']['std']

        # build discrete‐value → ID maps, reserve last idx for “unknown”
        self.discrete_labels = set(discrete_label_categorical_values.keys())
        self.val2id = {}
        for lbl, cats in discrete_label_categorical_values.items():
            m = {cat: i for i,cat in enumerate(cats)}
            m['<UNK>'] = len(cats)
            self.val2id[lbl] = m

    def tokenize(self, batch: list[dict]) -> dict[str,torch.Tensor]:
        B = len(batch)
        L = max((len(x['label']) if x else 0) for x in batch)
        # initialize
        labels_pad   = torch.full((B, L), self.pad_idx, dtype=torch.long)
        values_pad   = torch.zeros((B, L), dtype=torch.float32)
        vid_pad      = torch.zeros((B, L), dtype=torch.long)
        hours_pad    = torch.zeros((B, L), dtype=torch.float32)
        mask_pad     = torch.ones ((B, L), dtype=torch.bool)

        for i, rec in enumerate(batch):
            if not rec:
                continue
            lbls   = rec['label']
            vals   = rec.get('value', [])
            hrs    = rec.get('hours_from_intime', [])
            length = len(lbls)
            mask_pad[i, :length] = False

            # labels → indices
            labels_pad[i, :length] = torch.tensor(
                [ self.label_vocab.get(l, self.pad_idx) for l in lbls ],
                dtype=torch.long
            )
            # hours → normalize
            hours_pad[i, :length] = torch.tensor([
                (h - self.hrs_mean) / (self.hrs_std + 1e-6)
                for h in hrs
            ], dtype=torch.float32)

            # values → either continuous norm or categorical ID
            for j, l in enumerate(lbls):
                if l in self.discrete_labels:
                    # categorical
                    vid = self.val2id[l].get(vals[j], self.val2id[l]['<UNK>'])
                    vid_pad[i,j] = vid
                else:
                    # continuous
                    stats = self.val_stats.get(l)
                    if stats:
                        mean = stats['mean']
                        std  = stats['std']
                        values_pad[i,j] = (float(vals[j]) - mean) / (std + 1e-6)
                    else:
                        # fallback: leave at zero
                        values_pad[i,j] = 0.0

        return {
            'label'   : labels_pad,    # long
            'value'   : values_pad,    # float
            'value_id': vid_pad,       # long
            'hours'   : hours_pad,     # float
            'mask'    : mask_pad       # bool
        }
