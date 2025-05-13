import torch
from torch.nn.utils.rnn import pad_sequence

class MultimodalCollate:
    def __init__(self, tokenizer, modalities, max_len=512):
        """
        tokenizer: HF-style tokenizer
        modalities: same list you passed to the Dataset
        """
        self.tokenizer  = tokenizer
        self.modalities = modalities
        self.max_len    = max_len

    def __call__(self, batch):
        # batch is a list of (out_dict, flags_dict, label_tensor)
        outs_list, flags_list, labels_list = zip(*batch)

        # 1) Sequence modalities (anything except 'text')
        seq_data = {}
        seq_masks = {}
        for mod in self.modalities:
            if mod == 'text':
                continue

            # infer feature‐dim
            dim = next((outs[mod].shape[1] for outs in outs_list if outs[mod] is not None), None)
            seqs = [
                outs[mod] if outs[mod] is not None 
                else torch.zeros((1, dim))
                for outs in outs_list
            ]
            padded = pad_sequence(seqs, batch_first=True).float()
            lengths = torch.tensor([s.size(0) for s in seqs], device=padded.device)
            maxl = lengths.max()
            att  = torch.arange(maxl, device=lengths.device)[None, :] < lengths[:, None]

            seq_data[f"{mod}_pad"]   = padded
            seq_masks[mod] = ~att    # True where padded

        # 2) Text
        tokenized = None
        if 'text' in self.modalities:
            texts = [outs['text'] for outs in outs_list]
            tokenized = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )

        # 3) Missing‐modality mask: [B, num_modalities]
        miss_mask = torch.stack([
            (~torch.tensor([f[f"{mod}_missing"] for f in flags_list], dtype=torch.bool))
            for mod in self.modalities
        ], dim=1).float()

        # 4) Labels
        labels = torch.stack(labels_list, dim=0)

        # 5) Package everything into one dict for clarity
        batch_dict = { **seq_data }
        if tokenized is not None:
            batch_dict['text'] = tokenized

        return batch_dict, seq_masks, miss_mask, labels
