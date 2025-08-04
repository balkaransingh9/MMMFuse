import torch
import numpy as np

def prepare_embedding_batch(batch, num_notes=5):
    """
    Pads and batches pre-computed embeddings and their timestamps.
    This function can be used as a collate_fn in a DataLoader.

    Args:
        batch (list): A list of dictionaries from your dataset.
        num_notes (int): The fixed number of notes to pad/truncate to.

    Returns:
        A tuple of tensors: (padded_embeddings, padded_times, time_masks)
    """
    padded_embeddings_list = []
    padded_times_list = []
    time_masks_list = []

    for sample in batch:
        embeddings = torch.from_numpy(sample['embeddings'])
        times = sample['hours_from_intime']

        embeddings = embeddings[:num_notes]
        times = times[:num_notes]

        num_real_notes = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
        
        num_padding = num_notes - num_real_notes
        if num_padding > 0:
            pad_tensor = torch.zeros(num_padding, embedding_dim, dtype=torch.float32)
            final_embeddings = torch.cat([embeddings, pad_tensor], dim=0)
        else:
            final_embeddings = embeddings
            
        final_times = times + [-1.0] * num_padding
        time_mask = [1] * num_real_notes + [0] * num_padding

        padded_embeddings_list.append(final_embeddings)
        padded_times_list.append(torch.tensor(final_times, dtype=torch.float32))
        time_masks_list.append(torch.tensor(time_mask, dtype=torch.long))

    return (
        torch.stack(padded_embeddings_list),
        torch.stack(padded_times_list),
        torch.stack(time_masks_list)
    )