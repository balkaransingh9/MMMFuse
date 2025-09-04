import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

class ECGPatchTokenizer(nn.Module):
    def __init__(self, in_ch=12, embed_dim=128, patch_size=50, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size
        self.conv = nn.Conv1d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, bias=True)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("ECG should be a 3D tensor (B, 12, L) or (B, L, 12)")
        if x.shape[1] != 12 and x.shape[2] == 12:
            x = x.transpose(1, 2).contiguous()   # -> (B, 12, L)

        z = self.conv(x)                         # (B, D, T)
        z = z.transpose(1, 2).contiguous()       # (B, T, D)
        return z.transpose(0, 1).contiguous()    # (T, B, D)

class ECGSpikingEncoder(nn.Module):
    def __init__(self, in_ch=12, time_len=5000, patch_size=50, stride=None, 
                 embed_dim=128, hidden1=128, hidden2=128, 
                 tau1=2.0, tau2=2.0, p_drop=0.1, out_dim=1):
        super().__init__()

        self.patcher = ECGPatchTokenizer(in_ch=in_ch, embed_dim=embed_dim, 
                                         patch_size=patch_size, stride=stride)

        self.fc1  = layer.Linear(embed_dim, hidden1, bias=True)
        self.lif1 = neuron.ParametricLIFNode(init_tau=tau1, surrogate_function=neuron.surrogate.ATan(alpha=2.0),
                                             detach_reset=False, v_threshold=1.0, v_reset=None)

        self.drop = nn.Dropout(p_drop)

        self.fc2  = layer.Linear(hidden1, hidden2, bias=True)
        self.lif2 = neuron.ParametricLIFNode(init_tau=tau2, surrogate_function=neuron.surrogate.ATan(alpha=2.0),
                                             detach_reset=False, v_threshold=1.0, v_reset=None)

        self.readout = nn.Linear(hidden2, out_dim, bias=True)

        functional.set_step_mode(self, 'm')
        functional.set_backend(self, 'cupy')

    def forward(self, x, return_spikes = False, return_rates=False):
        # x = inputs['ecg']
        steps = self.patcher(x)          # (T, B, D)
        functional.reset_net(self)       # reset spiking states

        h1 = self.fc1(steps)             # (T, B, H1)
        s1 = self.lif1(h1)               # (T, B, H1)
        s1 = self.drop(s1)

        h2 = self.fc2(s1)                # (T, B, H2)
        s2 = self.lif2(h2)               # (T, B, H2)

        z = s2.mean(dim=0)               # temporal average (B, H2)
        # return self.readout(z)           # (B, out_dim)
        if return_spikes:
            if return_rates:
                return z, [s1.mean, s2.mean()]
            else:
                return z, {"s1":s1, "s2":s2}
        else:
            return z