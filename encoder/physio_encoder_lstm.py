import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class EHR_LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        nlayers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        return_cls: bool = True
    ):
        super().__init__()
        self.return_cls = return_cls
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.model_dim = hidden_dim * self.num_directions

        # project to hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=nlayers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if nlayers > 1 else 0.0
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        x = self.in_proj(input['pad']) * math.sqrt(self.hidden_dim)
        mask = input['attention_mask']

        # compute lengths; any zero-length sequences become length=1
        lengths = mask.sum(dim=1).cpu()
        lengths = lengths.clamp(min=1)

        # pack with enforce_sorted=False (you already have that)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)

        if self.return_cls:
            # extract the last hidden state
            # h_n shape: [num_layers * num_directions, B, hidden_dim]
            num_dirs = 2 if self.lstm.bidirectional else 1
            # take last layer
            last_layer = h_n.view(self.lstm.num_layers, num_dirs, x.size(0), self.hidden_dim)[-1]
            # concat directions
            h_final = torch.cat([last_layer[i] for i in range(num_dirs)], dim=-1)
            return self.dropout(h_final)
        else:
            # if you also want the full sequence output:
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            return self.dropout(out)