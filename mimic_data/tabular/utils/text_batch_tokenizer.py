import torch
from transformers import AutoTokenizer

def process_text_batch_with_mask(batch_data, tokenizer, max_length=512, num_notes=5):
    """
    Processes a batch of raw text data into tensors for the model,
    using -1 for time padding and creating a time mask.
    """
    all_input_ids = []
    all_attn_masks = []
    all_note_times = []
    all_note_time_masks = []

    for patient_data in batch_data:
        patient_notes = patient_data['text'][:num_notes]
        patient_times = patient_data['hours_from_intime'][:num_notes]

        inputs = tokenizer(
            patient_notes,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        num_real_notes = inputs['input_ids'].shape[0]
        num_padding_notes = num_notes - num_real_notes

        # Pad notes if necessary
        if num_padding_notes > 0:
            pad_ids = torch.zeros((num_padding_notes, max_length), dtype=torch.long)
            pad_mask = torch.zeros((num_padding_notes, max_length), dtype=torch.long)
            final_input_ids = torch.cat([inputs['input_ids'], pad_ids], dim=0)
            final_attn_mask = torch.cat([inputs['attention_mask'], pad_mask], dim=0)
        else:
            final_input_ids = inputs['input_ids']
            final_attn_mask = inputs['attention_mask']

        # 1. Pad timestamps with -1
        padded_times = patient_times + [-1.0] * num_padding_notes

        # 2. Create the corresponding time mask
        time_mask = [1] * num_real_notes + [0] * num_padding_notes

        all_input_ids.append(final_input_ids)
        all_attn_masks.append(final_attn_mask)
        all_note_times.append(torch.tensor(padded_times, dtype=torch.float))
        all_note_time_masks.append(torch.tensor(time_mask, dtype=torch.long))

    return {
        'inputs_ids': torch.stack(all_input_ids),
        'attention_masks': torch.stack(all_attn_masks),
        'note_times': torch.stack(all_note_times),
        'note_time_masks': torch.stack(all_note_time_masks) # Return the new mask
    }