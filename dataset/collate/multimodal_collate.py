import torch
from torch.nn.utils.rnn import pad_sequence

class MultimodalCollate:
    def __init__(self, tokenizer, max_len=512, task_type='phenotype'):
        """
        Collate class for multimodal data (physio, ECG, text).
        
        Args:
            tokenizer: HuggingFace-style tokenizer with __call__ method.
            max_len (int): Maximum sequence length for text tokenization.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if task_type == 'phenotype':
            self.task_type = task_type
        elif task_type == 'in_hospital_mortality':
            self.task_type = task_type
        elif task_type == 'length_of_stay':
            self.task_type = task_type
        else:
            raise ValueError("Unsupported task type!")

    def __call__(self, batch):
        physio_list, ecg_list, text_list, missing_flag_list, labels_list = zip(*batch)

        # === Physio data ===
        physio_dim = next((x.shape[1] for x in physio_list if x is not None), None)
        physio_seq = [x if x is not None else torch.zeros((1, physio_dim)) for x in physio_list]
        physio_pad = pad_sequence(physio_seq, batch_first=True).float()
        physio_lengths = torch.tensor([seq.size(0) for seq in physio_seq])
        physio_max_len = physio_lengths.max()
        physio_attention_mask = torch.arange(physio_max_len,
                                             device=physio_lengths.device)[None, :] < physio_lengths[:, None]

        # === ECG data ===
        ecg_dim = next((x.shape[1] for x in ecg_list if x is not None), None)
        ecg_seq = [x if x is not None else torch.zeros((1, ecg_dim)) for x in ecg_list]
        ecg_pad = pad_sequence(ecg_seq, batch_first=True).float()
        ecg_lengths = torch.tensor([seq.size(0) for seq in ecg_seq])
        ecg_max_len = ecg_lengths.max()
        ecg_attention_mask = torch.arange(ecg_max_len,
                                          device=ecg_lengths.device)[None, :] < ecg_lengths[:, None]

        # === Text data ===
        tokenized_text = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # === Masks ===
        attention_masks = {'physio': ~physio_attention_mask, 'ecg': ~ecg_attention_mask}

        missing_mod_mask = torch.stack([
            (~torch.tensor([f['physio_missing'] for f in missing_flag_list], dtype=torch.bool)),
            (~torch.tensor([f['ecg_missing']    for f in missing_flag_list], dtype=torch.bool)),
            (~torch.tensor([f['text_missing']   for f in missing_flag_list], dtype=torch.bool)),
        ], dim=1).float()  # [B, 3]

        # === Labels ===
        if self.task_type == 'phenotype':
            labels = torch.stack(labels_list, dim=0)
        elif self.task_type == 'in_hospital_mortality':
            labels = torch.tensor(labels_list).unsqueeze(1)
        else:
            labels = torch.tensor(labels_list).long()

        return physio_pad, ecg_pad, tokenized_text, attention_masks, missing_mod_mask, labels