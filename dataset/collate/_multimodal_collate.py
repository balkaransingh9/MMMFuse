import torch
from torch.nn.utils.rnn import pad_sequence

class MultimodalCollate:
    def __init__(
        self,
        modalities,
        text_tokenizer,
        med_tokenizer,
        task_type='phenotype',
        text_max_len=512,
        med_max_len=128,
        text_kwargs=None,
        med_kwargs=None
        ):
        """
        Args:
            text_tokenizer: HF-style tokenizer for text modality
            med_tokenizer: custom tokenizer for med modality
            modalities: list of modality names, e.g. ['text','med','lab','ecg',…]
            text_max_len: max_length for text tokenizer
            med_max_len: max_length (or other limit) for med tokenizer
            text_kwargs / med_kwargs: additional dicts of tokenizer args
            task_type: one of ['phenotype', 'in_hospital_mortality', 'length_of_stay']
        """

        if task_type not in ['phenotype', 'in_hospital_mortality', 'length_of_stay']:
            raise ValueError(f"Unsupported task type: {task_type}")
        self.task_type = task_type

        self.modalities     = modalities
        self.text_tok       = text_tokenizer
        self.med_tok        = med_tokenizer
        self.text_max_len   = text_max_len
        self.med_max_len    = med_max_len
        self.text_kwargs    = text_kwargs or {}
        self.med_kwargs     = med_kwargs  or {}

    def __call__(self, batch):
        outs_list, flags_list, labels_list = zip(*batch)

        # 1) Sequence modalities
        seq_data  = {}
        seq_masks = {}
        for mod in self.modalities:
            if mod in ('text', 'med'):
                continue

            dim = next((o[mod].shape[1] for o in outs_list if o[mod] is not None), None)
            seqs = [
                o[mod] if o[mod] is not None else torch.zeros((1, dim))
                for o in outs_list
            ]
            padded = pad_sequence(seqs, batch_first=True).float()
            lengths = torch.tensor([s.size(0) for s in seqs], device=padded.device)
            maxl    = lengths.max()
            att     = torch.arange(maxl, device=lengths.device)[None, :] < lengths[:, None]

            seq_data[f"{mod}_pad"] = padded
            seq_masks[mod]         = ~att

        # 2) Text modality
        if 'text' in self.modalities:
            texts = [o['text'] for o in outs_list]
            tokenized = self.text_tok(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.text_max_len,
                return_tensors='pt',
                **self.text_kwargs
            )
            seq_data['text'] = tokenized

        # 3) Med modality
        if 'med' in self.modalities:
            meds = [o['med'] for o in outs_list]
            med_tokenized = self.med_tok.tokenize(meds)
            seq_data['med'] = med_tokenized

        # 4) Missing-modality mask
        miss_mask = torch.stack([
            (~torch.tensor([f.get(f"{mod}_missing", True) for f in flags_list], dtype=torch.bool))
            for mod in self.modalities
        ], dim=1).float()

        # 5) Labels
        if self.task_type == 'phenotype':
            labels = torch.stack(labels_list, dim=0)
        elif self.task_type == 'in_hospital_mortality':
            labels = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
        else:  # 'length_of_stay'
            labels = torch.tensor(labels_list, dtype=torch.long)

        return seq_data, seq_masks, miss_mask, labels