import torch
from torch.nn.utils.rnn import pad_sequence

class PhysioCollate:
    def __init__(self,  task_type='phentype'):
        """
        Collate function class for physiological data only.
        """
        if task_type == 'phentype':
            self.task_type = task_type
        elif task_type == 'in_hospital_mortality':
            self.task_type = task_type
        elif task_type == 'length_of_stay':
            self.task_type = task_type
        else:
            raise ValueError("Unsupported task type!")

    def __call__(self, batch):
        physio_list, labels_list = zip(*batch)

        # Determine feature dimension
        physio_dim = next((x.shape[1] for x in physio_list if x is not None), None)

        # Fill missing sequences with zero-padded tensors
        physio_seq = [x if x is not None else torch.zeros((1, physio_dim)) for x in physio_list]

        # Pad sequences
        physio_pad = pad_sequence(physio_seq, batch_first=True).float()
        physio_lengths = torch.tensor([seq.size(0) for seq in physio_seq])
        physio_max_len = physio_lengths.max()

        # Create attention mask (True = valid, False = padding)
        physio_attention_mask = torch.arange(physio_max_len,
                                             device=physio_lengths.device)[None, :] < physio_lengths[:, None]

        if self.task_type == 'length_of_stay':
            labels = torch.tensor(labels_list)
        else:
            labels = torch.stack(labels_list, dim=0)

        attention_masks = {'physio': ~physio_attention_mask}
        #return {"x_data": physio_pad, "attention_masks": attention_masks, "labels": labels}
        return {"x": physio_pad, "mask": ~physio_attention_mask, "labels": labels}