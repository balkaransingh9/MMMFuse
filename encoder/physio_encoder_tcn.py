import torch
import torch.nn as nn
import math

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size,
                               padding=(kernel_size-1)*dilation,
                               dilation=dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size,
                               padding=(kernel_size-1)*dilation,
                               dilation=dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # match residual dimensions if needed
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        # crop to original length if padding was excessive
        return out[..., :x.size(2)] + res

class EHR_TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: list = [64, 64, 128],
        kernel_size: int = 3,
        dropout: float = 0.2,
        return_cls: bool = True
    ):
        """
        input_dim: number of features per time step
        num_channels: list of output channels for each TCN block
        kernel_size: convolution kernel size
        dropout: dropout after each conv
        return_cls: if True, returns [B, C_last] via global avg pool; else [B, T, C_last]
        """
        super().__init__()
        layers = []
        in_c = input_dim
        for out_c in num_channels:
            layers.append(TemporalBlock(in_c, out_c, kernel_size,
                                        dilation=2**len(layers),
                                        dropout=dropout))
            in_c = out_c
        self.network = nn.Sequential(*layers)
        self.return_cls = return_cls
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.model_dim = num_channels[-1]

    def forward(self, input):
        # input['pad']: [B, T, D]
        # input['attention_mask']: [B, T]  (not used by TCN)
        x = input['pad'].transpose(1, 2)      # → [B, D, T]
        y = self.network(x)                  # [B, C_last, T]
        if self.return_cls:
            # global average over time dimension
            out = self.global_pool(y).squeeze(-1)  # [B, C_last]
            return out
        else:
            return y.transpose(1, 2)         # [B, T, C_last]