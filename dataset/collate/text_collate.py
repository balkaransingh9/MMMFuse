import torch

class TextCollate:
    def __init__(self, tokenizer, max_len=250, task_type='phenotype'):
        """
        Collate class for tokenizing text data and batching labels.

        Args:
            tokenizer: HuggingFace-style tokenizer.
            max_len (int): Maximum sequence length for tokenized text.
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
        text_list, labels_list = zip(*batch)

        # Tokenize the batch of texts
        tokenized_text = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        if self.task_type == 'phenotype':
            labels = torch.stack(labels_list, dim=0)
        elif self.task_type == 'in_hospital_mortality':
            labels = torch.tensor(labels_list).unsqueeze(1)
        else:
            labels = torch.tensor(labels_list).long()

        return {"x_data":tokenized_text, "labels":labels}