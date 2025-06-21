import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name, model_dim=128, dropout_rate=0.2, output_type='mean'):
        """
        output_type: one of ['cls', 'mean', 'token']
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.common_projection = nn.Linear(hidden_size, model_dim)
        self.dropout = nn.Dropout(dropout_rate)

        assert output_type in ['cls', 'mean', 'token'], "output_type must be 'cls', 'mean', or 'token'"
        self.output_type = output_type

    def mean_pooling(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        masked_hidden = last_hidden * mask
        summed = masked_hidden.sum(1)
        count = mask.sum(1)
        return summed / count

    def forward(self, input):
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.output_type == 'cls':
            out = outputs.pooler_output
            if out is None:
                raise ValueError("Model does not return pooler_output. Use output_type='mean' instead.")
            out = self.common_projection(out)
            out = self.dropout(out)
            return out

        elif self.output_type == 'mean':
            last_hidden = outputs.last_hidden_state
            out = self.mean_pooling(last_hidden, attention_mask)
            out = self.common_projection(out)
            out = self.dropout(out)
            return out

        elif self.output_type == 'token':
            return outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim] 