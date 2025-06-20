import torch
import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name, model_dim=128, dropout_rate=0.2, return_cls=True):
      super().__init__()
      self.encoder = AutoModel.from_pretrained(model_name)
      hidden_size = self.encoder.config.hidden_size
      self.common_projection = nn.Linear(hidden_size, model_dim)
      self.dropout = nn.Dropout(dropout_rate)
      self.return_cls = return_cls

    def forward(self, input):
      input_ids = input['input_ids']
      attention_mask = input['attention_mask']

      outputs = self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask)
      
  
      if self.return_cls == True:
         out = outputs.pooler_output
         out = self.common_projection(out)
         out = self.dropout(out)
         return out
      else:
         out = outputs.last_hidden_state
         return out    
    