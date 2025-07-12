import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
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
        # input['pad']: [B, T, input_dim]
        # input['attention_mask']: [B, T]  where 1=real, 0=pad
        x = self.in_proj(input['pad']) * math.sqrt(self.hidden_dim)

        # infer lengths from mask
        lengths = input['attention_mask'].sum(dim=1).cpu()
        # pack
        packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        # h_n: [num_layers * num_directions, B, hidden_dim]

        if self.return_cls:
            # take the last layer's hidden state for each direction and concat
            h_final = h_n.view(self.lstm.num_layers, 
                               self.num_directions,
                               x.size(0),
                               self.hidden_dim)[-1]  # last layer  
            # shape: [num_directions, B, hidden_dim]
            h_final = torch.cat([h_final[i] for i in range(self.num_directions)], dim=-1)
            # shape: [B, hidden_dim * num_directions]
            return self.dropout(h_final)
        else:
            # unpack if you really need the sequence output
            out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
            return self.dropout(out)