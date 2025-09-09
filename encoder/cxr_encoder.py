import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple, DropPath

from spikingjelly.activation_based import layer as abL
from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode
from spikingjelly.activation_based import functional as abF

# ---------- helpers ----------
def _make_neuron(spike_mode: str, *, tau=2.0, v_threshold=0.5, step_mode='m', detach_reset=True):
    if spike_mode == "lif":
        return LIFNode(tau=tau, v_threshold=v_threshold, detach_reset=detach_reset, step_mode=step_mode)
    elif spike_mode == "plif":
        return ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, detach_reset=detach_reset, step_mode=step_mode)
    else:
        raise ValueError(f"Unsupported spike_mode: {spike_mode}")

def _save_hook(hook, name, x, hook_grad: bool):
    if hook is not None:
        hook[name] = x if hook_grad else x.detach()

# ---------- layers ----------
class Erode(nn.Module):
    """MaxPool3d over spatial dims while preserving time. I/O: [T,B,C,H,W]"""
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))

    def forward(self, x):
        # [T,B,C,H,W] -> [B,C,T,H,W] -> pool -> [T,B,C,H,W]
        x = x.permute(1,2,0,3,4).contiguous()
        x = self.pool(x)
        x = x.permute(2,0,1,3,4).contiguous()
        return x

class MS_MLP_Conv(nn.Module):
    """
    Time-aware MLP with 1x1 convs, activation_based. I/O: [T,B,C,H,W]
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.0, spike_mode="lif", layer=0):
        super().__init__()
        out_features  = out_features  or in_features
        hidden_features = hidden_features or in_features
        self.res    = in_features == hidden_features
        self.layer  = layer
        self.c_hidden = hidden_features
        self.c_output = out_features

        self.fc1_conv = abL.Conv2d(in_features, hidden_features, 1, 1, bias=True)
        self.fc1_bn   = abL.BatchNorm2d(hidden_features)
        self.fc1_lif  = _make_neuron(spike_mode, tau=2.0, step_mode='m')

        self.fc2_conv = abL.Conv2d(hidden_features, out_features, 1, 1, bias=True)
        self.fc2_bn   = abL.BatchNorm2d(out_features)
        self.fc2_lif  = _make_neuron(spike_mode, tau=2.0, step_mode='m')

    def forward(self, x, hook=None, hook_grad: bool=False):
        identity = x
        x = self.fc1_conv(x); x = self.fc1_bn(x); x = self.fc1_lif(x)
        _save_hook(hook, self._get_name()+str(self.layer)+"_fc1_lif", x, hook_grad)

        if self.res:
            x = x + identity
            identity = x

        x = self.fc2_conv(x); x = self.fc2_bn(x); x = self.fc2_lif(x)
        _save_hook(hook, self._get_name()+str(self.layer)+"_fc2_lif", x, hook_grad)

        x = x + identity
        return x, hook

class MS_SSA_Conv(nn.Module):
    """
    Spiking self-attention-ish block using multiplicative q*(k*v).
    All convs/BNs are activation_based for [T,B,C,H,W].
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, sr_ratio=1, mode="direct_xor",
                 spike_mode="lif", dvs=False, layer=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.dvs = dvs
        self.layer = layer
        self.mode = mode

        if dvs:
            self.pool3d = Erode()

        self.q_conv = abL.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.q_bn   = abL.BatchNorm2d(dim)
        self.q_lif  = _make_neuron(spike_mode, tau=2.0, step_mode='m')

        self.k_conv = abL.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.k_bn   = abL.BatchNorm2d(dim)
        self.k_lif  = _make_neuron(spike_mode, tau=2.0, step_mode='m')

        self.v_conv = abL.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.v_bn   = abL.BatchNorm2d(dim)
        self.v_lif  = _make_neuron(spike_mode, tau=2.0, step_mode='m')

        self.talking_heads     = nn.Conv1d(num_heads, num_heads, 1, 1, bias=False)
        self.talking_heads_lif = _make_neuron(spike_mode, tau=2.0, v_threshold=0.5, step_mode='m')

        self.proj_conv = abL.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_bn   = abL.BatchNorm2d(dim)

        self.shortcut_lif = _make_neuron(spike_mode, tau=2.0, step_mode='m')

    def forward(self, x, hook=None, hook_grad: bool=False):
        identity = x
        T,B,C,H,W = x.shape
        N = H * W

        x = self.shortcut_lif(x)
        _save_hook(hook, self._get_name()+str(self.layer)+"_first_lif", x, hook_grad)

        q = self.q_lif(self.q_bn(self.q_conv(x)))   # [T,B,C,H,W]
        _save_hook(hook, self._get_name()+str(self.layer)+"_q_lif", q, hook_grad)
        q = (q.flatten(3).transpose(-1,-2)
               .reshape(T,B,N,self.num_heads,C//self.num_heads)
               .permute(0,1,3,2,4).contiguous())   # [T,B,heads,N,Ch]

        k = self.k_lif(self.k_bn(self.k_conv(x)))
        if self.dvs: k = self.pool3d(k)
        _save_hook(hook, self._get_name()+str(self.layer)+"_k_lif", k, hook_grad)
        k = (k.flatten(3).transpose(-1,-2)
               .reshape(T,B,N,self.num_heads,C//self.num_heads)
               .permute(0,1,3,2,4).contiguous())

        v = self.v_lif(self.v_bn(self.v_conv(x)))
        if self.dvs: v = self.pool3d(v)
        _save_hook(hook, self._get_name()+str(self.layer)+"_v_lif", v, hook_grad)
        v = (v.flatten(3).transpose(-1,-2)
               .reshape(T,B,N,self.num_heads,C//self.num_heads)
               .permute(0,1,3,2,4).contiguous())

        kv = k.mul(v)                        # [T,B,heads,N,Ch]
        _save_hook(hook, self._get_name()+str(self.layer)+"_kv_before", kv, hook_grad)

        kv = kv.sum(dim=-2, keepdim=True)    # [T,B,heads,1,Ch]
        kv_1d = kv.view(T*B, self.num_heads, 1, -1).squeeze(2)  # [T*B, heads, Ch]
        kv_1d = self.talking_heads(kv_1d)                       # [T*B, heads, Ch]
        kv = kv_1d.view(T,B,self.num_heads,1,-1)
        kv = self.talking_heads_lif(kv)
        _save_hook(hook, self._get_name()+str(self.layer)+"_kv", kv, hook_grad)

        x_att = q.mul(kv)                    # [T,B,heads,N,Ch]
        _save_hook(hook, self._get_name()+str(self.layer)+"_x_after_qkv", x_att, hook_grad)

        x_att = x_att.transpose(3,4).reshape(T,B,C,H,W).contiguous()
        x_out = self.proj_bn(self.proj_conv(x_att))
        x_out = x_out + identity
        return x_out, v, hook

class MS_Block_Conv(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, sr_ratio=1, attn_mode="direct_xor",
                 spike_mode="lif", dvs=False, layer=0):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
            mode=attn_mode, spike_mode=spike_mode, dvs=dvs, layer=layer
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(in_features=dim, hidden_features=mlp_hidden_dim,
                               drop=drop, spike_mode=spike_mode, layer=layer)

    def forward(self, x, hook=None, hook_grad: bool=False):
        x_attn, attn_feat, hook = self.attn(x, hook=hook, hook_grad=hook_grad)
        x, hook = self.mlp(x_attn, hook=hook, hook_grad=hook_grad)
        return x, attn_feat, hook

class MS_SPS(nn.Module):
    """
    Spiking Patch/Stem with hierarchical downsampling. I/O: [T,B,C,H,W]
    """
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4,
                 in_channels=2, embed_dims=256, pooling_stat="1111", spike_mode="lif"):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat
        self.C = in_channels

        # stage 0
        self.proj_conv = abL.Conv2d(in_channels, embed_dims // 8, 3, 1, 1, bias=False)
        self.proj_bn   = abL.BatchNorm2d(embed_dims // 8)
        self.proj_lif  = _make_neuron(spike_mode, tau=2.0, step_mode='m')
        self.maxpool   = abL.MaxPool2d(3, 2, 1)

        # stage 1
        self.proj_conv1 = abL.Conv2d(embed_dims // 8, embed_dims // 4, 3, 1, 1, bias=False)
        self.proj_bn1   = abL.BatchNorm2d(embed_dims // 4)
        self.proj_lif1  = _make_neuron(spike_mode, tau=2.0, step_mode='m')
        self.maxpool1   = abL.MaxPool2d(3, 2, 1)

        # stage 2
        self.proj_conv2 = abL.Conv2d(embed_dims // 4, embed_dims // 2, 3, 1, 1, bias=False)
        self.proj_bn2   = abL.BatchNorm2d(embed_dims // 2)
        self.proj_lif2  = _make_neuron(spike_mode, tau=2.0, step_mode='m')
        self.maxpool2   = abL.MaxPool2d(3, 2, 1)

        # stage 3
        self.proj_conv3 = abL.Conv2d(embed_dims // 2, embed_dims, 3, 1, 1, bias=False)
        self.proj_bn3   = abL.BatchNorm2d(embed_dims)
        self.proj_lif3  = _make_neuron(spike_mode, tau=2.0, step_mode='m')
        self.maxpool3   = abL.MaxPool2d(3, 2, 1)

        # relative position conv
        self.rpe_conv = abL.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False)
        self.rpe_bn   = abL.BatchNorm2d(embed_dims)
        self.rpe_lif  = _make_neuron(spike_mode, tau=2.0, step_mode='m')

    def forward(self, x, hook=None, hook_grad: bool=False):
        # stage 0
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)))
        _save_hook(hook, self._get_name()+"_lif", x, hook_grad)
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)

        # stage 1
        x = self.proj_lif1(self.proj_bn1(self.proj_conv1(x)))
        _save_hook(hook, self._get_name()+"_lif1", x, hook_grad)
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)

        # stage 2
        x = self.proj_lif2(self.proj_bn2(self.proj_conv2(x)))
        _save_hook(hook, self._get_name()+"_lif2", x, hook_grad)
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)

        # stage 3 (+ RPE)
        x_feat = self.proj_bn3(self.proj_conv3(x))      # no spike yet
        if self.pooling_stat[3] == "1":
            x_feat = self.maxpool3(x_feat)

        x = self.proj_lif3(x_feat)
        _save_hook(hook, self._get_name()+"_lif3", x, hook_grad)

        rpe = self.rpe_bn(self.rpe_conv(x))
        x = x + rpe

        T,B,C,H,W = x.shape
        H_out = H // self.patch_size[0]
        W_out = W // self.patch_size[1]
        return x, (H_out, W_out), hook

# ---------- model ----------
class SpikeDrivenTransformer(nn.Module):
    """
    Single-stage stack of MS_Block_Conv blocks with grad-capable hooks.
    """
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=512, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm,
                 depths=12, sr_ratios=4, T=4, pooling_stat="1111", attn_mode="direct_xor",
                 spike_mode="lif", get_embed=False, dvs_mode=False, TET=False, cml=False,
                 pretrained=False, pretrained_cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths if isinstance(depths, int) else int(sum(depths))
        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        # stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(self.depths,1))] if self.depths > 0 else []

        # stem / patch embed
        self.patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        # transformer-like flat stack
        self.block = nn.ModuleList([
            MS_Block_Conv(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[j] if j < len(dpr) else 0.0,
                norm_layer=norm_layer,
                sr_ratio=sr_ratios if isinstance(sr_ratios, int) else sr_ratios[0],
                attn_mode=attn_mode,
                spike_mode=spike_mode,
                dvs=dvs_mode,
                layer=j,
            )
            for j in range(self.depths)
        ])

        # classification head
        self.head_lif = _make_neuron(spike_mode, tau=2.0, step_mode='m')
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        abF.set_step_mode(self, 'm')
        abF.set_backend(self, 'cupy')

    def _init_weights(self, m):
        if isinstance(m, (abL.Conv2d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (abL.BatchNorm2d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None, hook_grad: bool=False):
        x, _, hook = self.patch_embed(x, hook=hook, hook_grad=hook_grad)
        for blk in self.block:
            x, _, hook = blk(x, hook=hook, hook_grad=hook_grad)
        # global average over spatial dims
        x = x.flatten(3).mean(3)  # [T,B,C]
        return x, hook

    def forward(self, x, hook=None, hook_grad: bool=False):
        """
        x can be:
          - [B,C,H,W] -> replicated to [T,B,C,H,W]
          - [T,B,C,H,W] (time-first)
        """
        # reset all neuron states for a fresh sample
        abF.reset_net(self)

        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        else:
            assert x.dim() == 5, f"Expected 4D or 5D input, got {x.shape}"

        x, hook = self.forward_features(x, hook=hook, hook_grad=hook_grad)
        x = self.head_lif(x)               # [T,B,C]
        _save_hook(hook, "head_lif", x, hook_grad)

        x = self.head(x)                   # [T,B,num_classes]
        if not self.TET:
            x = x.mean(0)                  # [B,num_classes]
        return x, hook

import re

def _unwrap_state_dict(sd):
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        return sd["state_dict"]
    return sd

def _strip_known_prefixes(sd, prefixes=("model.", "net.", "backbone.")):
    remapped = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        remapped[k] = v
    return remapped

def _auto_prefix_alignment(sd_keys):
    if not sd_keys:
        return None
    first = sd_keys[0]
    if "." in first:
        root = first.split(".")[0] + "."
        if all(k.startswith(root) for k in sd_keys):
            return root
    return None

def load_weights_safely(
    model,
    ckpt_path: str,
    drop_head: bool = True,
    head_key_regex: str = r"^head\.(weight|bias)$",
    verbose: bool = True,
):
    sd_raw = torch.load(ckpt_path, map_location="cpu")
    sd = _unwrap_state_dict(sd_raw)

    if drop_head:
        pat = re.compile(head_key_regex)
        sd = {k: v for k, v in sd.items() if not pat.match(k)}

    auto = _auto_prefix_alignment(list(sd.keys()))
    if auto:
        sd = {k[len(auto):]: v for k, v in sd.items()}

    sd = _strip_known_prefixes(sd, prefixes=("model.", "net.", "backbone."))

    model_sd = model.state_dict()
    ok, skipped_by_shape = {}, []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            ok[k] = v
        elif k in model_sd:
            skipped_by_shape.append(k)

    missing, unexpected = model.load_state_dict(ok, strict=False)

    if verbose:
        print(f"[safe-load] from: {ckpt_path}")
        print(f"  loaded params: {len(ok)}")
        print(f"  skipped by shape: {len(skipped_by_shape)}")
        if skipped_by_shape:
            print("  e.g.", skipped_by_shape[:8])
        print(f"  missing in ckpt (but in model): {len(missing)}")
        if missing:
            print("  e.g.", missing[:8])
        print(f"  unexpected in ckpt (no match in model): {len(unexpected)}")
        if unexpected:
            print("  e.g.", unexpected[:8])

    return missing, unexpected, skipped_by_shape

class SpikeImageFeats(nn.Module):
    def __init__(self,
                 backbone,             
                 ckpt_path: str = None,     
                 out_dim: int = None,       
                 stage: str = "prehead",    
                 time_reduce: str = "mean", 
                 drop_head: bool = True,    
                 verbose_load: bool = True):
        super().__init__()
        self.backbone = backbone
        self.stage = stage
        self.time_reduce = time_reduce

        if ckpt_path is not None:
            load_weights_safely(self.backbone, ckpt_path, drop_head=drop_head, verbose=verbose_load)

        in_dim = self.backbone.head.in_features
        if out_dim is None or out_dim == in_dim:
            self.proj = nn.Identity()
            self.out_dim = in_dim
        else:
            self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))
            self.out_dim = out_dim

    def forward(self, images: torch.Tensor, return_spikes: bool = False, return_rates = False):
        """
        images: (B,C,H,W) or (T,B,C,H,W)
        returns:
          if time_reduce != 'none': (B, out_dim)
          else:                     (T,B,out_dim)
        """
        abF.reset_net(self.backbone)

        if images.dim() == 4:
            x = images.unsqueeze(0).repeat(self.backbone.T, 1, 1, 1, 1)
        elif images.dim() == 5:
            x = images
        else:
            raise ValueError(f"Expected (B,C,H,W) or (T,B,C,H,W), got {tuple(images.shape)}")

        hook = {} if return_spikes else None
        feats, hook = self.backbone.forward_features(x, hook=hook, hook_grad=True)  


        if self.stage.lower() == "head_lif":
            feats = self.backbone.head_lif(feats)                 

        if self.time_reduce == "mean":
            feats = feats.mean(0)                                
        elif self.time_reduce == "sum":
            feats = feats.sum(0)
        elif self.time_reduce == "last":
            feats = feats[-1]
        elif self.time_reduce == "none":
            pass
        else:
            raise ValueError("time_reduce must be 'mean'|'sum'|'last'|'none'")

        if self.time_reduce == "none":
            T,B,D = feats.shape
            feats = self.proj(feats.view(T*B, D)).view(T, B, self.out_dim)
        else:
            feats = self.proj(feats)
        
        if return_spikes:
            
            if return_rates:
                lif_keys = [k for k in hook if k.endswith('_lif') or 'head_lif' in k]
                rates = []
                for k in lif_keys:
                    s = hook[k]             # retains grad
                    rates.append(s.float().mean())
                return feats, rates
        
        else:
            return feats
