import torch
from torch.nn.utils.rnn import pad_sequence

class ECGCollate:
    def __init__(self):
        """
        Collate class for ECG data batching.
        """
        pass

    def __call__(self, batch):
        ecg_list, labels_list = zip(*batch)

        # Determine the ECG feature dimension
        ecg_dim = next((x.shape[1] for x in ecg_list if x is not None), None)
        ecg_seq = [x if x is not None else torch.zeros((1, ecg_dim)) for x in ecg_list]

        # Pad the ECG sequences
        ecg_pad = pad_sequence(ecg_seq, batch_first=True).float()
        ecg_lengths = torch.tensor([seq.size(0) for seq in ecg_seq])
        ecg_max_len = ecg_lengths.max()

        # Generate attention mask (True = padding)
        ecg_attention_mask = torch.arange(ecg_max_len,
                                          device=ecg_lengths.device)[None, :] < ecg_lengths[:, None]

        attention_masks = {'ecg': ~ecg_attention_mask}

        return ecg_pad, attention_masks