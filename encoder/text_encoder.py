import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name, model_dim=128):
      super().__init__()
      self.encoder = AutoModel.from_pretrained(model_name)
      hidden_size = self.encoder.config.hidden_size
      self.common_projection = nn.Linear(hidden_size, model_dim)
      self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
      outputs = self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask)

      x = self.common_projection(outputs.last_hidden_state)
      return self.dropout(x)