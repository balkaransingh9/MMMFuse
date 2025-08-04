import torch
import numpy as np

def prepare_embedding_batch(batch, num_notes=5, embedding_dim=768):
    padded_embeddings_list = []
    padded_times_list = []
    time_masks_list = []

    default_embeddings = torch.zeros(num_notes, embedding_dim, dtype=torch.float32)
    default_times = torch.full((num_notes,), -1.0, dtype=torch.float32)
    default_mask = torch.zeros(num_notes, dtype=torch.long)

    if batch is None or not batch:
        return {
            'embeddings' : torch.empty(0, num_notes, embedding_dim),
            'note_times' : torch.empty(0, num_notes),
            'note_time_masks' : torch.empty(0, num_notes, dtype=torch.long)
        }

    for sample in batch:
        is_invalid = (
            sample is None or
            'embeddings' not in sample or
            sample['embeddings'].shape[0] == 0
        )

        if is_invalid:
            padded_embeddings_list.append(default_embeddings)
            padded_times_list.append(default_times)
            time_masks_list.append(default_mask)
            continue

        final_embeddings = torch.zeros(num_notes, embedding_dim, dtype=torch.float32)
        final_times = torch.full((num_notes,), -1.0, dtype=torch.float32)
        final_mask = torch.zeros(num_notes, dtype=torch.long)
        
        embeddings = torch.from_numpy(sample['embeddings'])
        times = torch.tensor(sample.get('hours_from_intime', []), dtype=torch.float32)
        
        normalized_times = torch.tensor([])
        if len(times) > 0:
            min_time = torch.min(times)
            max_time = torch.max(times)
            
            # Avoid division by zero if all timestamps are the same
            if max_time > min_time:
                # Min-max scale to the range [0, 48]
                normalized_times = (times - min_time) * 48.0 / (max_time - min_time)
            else:
                # If all times are the same, their normalized position is 0
                normalized_times = torch.zeros_like(times)

        num_to_copy = min(embeddings.shape[0], num_notes)
        
        final_embeddings[:num_to_copy] = embeddings[:num_to_copy]
        
        num_times_to_copy = min(len(normalized_times), num_to_copy)
        if num_times_to_copy > 0:
            final_times[:num_times_to_copy] = normalized_times[:num_times_to_copy]
        
        final_mask[:num_to_copy] = 1 # Set mask to 1 for real data

        padded_embeddings_list.append(final_embeddings)
        padded_times_list.append(final_times)
        time_masks_list.append(final_mask)

    return {
        'embeddings' : torch.stack(padded_embeddings_list),
        'note_times' : torch.stack(padded_times_list),
        'note_time_masks' : torch.stack(time_masks_list)
    }