import torch

class TextCollate:
    def __init__(self, tokenizer, max_len=250):
        """
        Collate class for tokenizing text data and batching labels.

        Args:
            tokenizer: HuggingFace-style tokenizer.
            max_len (int): Maximum sequence length for tokenized text.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        text_list, labels_list = zip(*batch)

        # Tokenize the batch of texts
        tokenized_text = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Stack labels into a single tensor
        labels = torch.stack(labels_list, dim=0)

        return {"x_data":tokenized_text, "labels":labels}