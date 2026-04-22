import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_
from mamba_ssm import Mamba

try:
    from .ops_dcnv3.functions import DCNv3Function
except ImportError:
    print("Warning: ops_dcnv3.functions.DCNv3Function not found. Ensure DCNv3 is compiled and in the python path.")
    DCNv3Function = None

# =========================================================================
# VAMamba Components (Macro-Level Pathfinding)
# =========================================================================

class CNNScoreMap(nn.Module):
    def __init__(self, in_channels, embed_dim=64):
        super().__init__()
        # Lightweight depthwise separable configuration
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, groups=in_channels if in_channels <= embed_dim else 1)
        self.act1 = nn.GELU()
        
        # Channel Attention (Squeeze-and-Excitation style)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc1 = nn.Conv2d(embed_dim, max(1, embed_dim // 4), 1, bias=False)
        self.ca_act = nn.ReLU(inplace=True)
        self.ca_fc2 = nn.Conv2d(max(1, embed_dim // 4), embed_dim, 1, bias=False)
        self.ca_sigmoid = nn.Sigmoid()

        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.act2 = nn.GELU()
        
        # Score projection
        self.score_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x):
        # x is [B, C, H, W]
        if x.dim() == 4 and x.shape[1] != self.conv1.in_channels:
            # Re-permute if it was passed as channels last somehow
            x = x.permute(0, 3, 1, 2).contiguous()
            
        feat = self.act1(self.conv1(x))
        
        # Apply Channel Attention
        ca = self.global_pool(feat)
        ca = self.ca_fc1(ca)
        ca = self.ca_act(ca)
        ca = self.ca_fc2(ca)
        ca = self.ca_sigmoid(ca)
        feat = feat * ca
        
        feat = self.act2(self.conv2(feat))
        scores = self.score_head(feat) # [B, 1, H, W]
        scores = torch.sigmoid(scores).squeeze(1) # [B, H, W]
        
        # Blend in local variance as a fast content heuristic
        patch_variance = torch.var(x, dim=1) # [B, H, W]
        content_score = torch.sigmoid(patch_variance * 10)
        
        final_scores = 0.7 * scores + 0.3 * content_score
        return final_scores


# =========================================================================
# DAMamba Components (Micro-Level Adaptive Masking)
# =========================================================================

class to_channels_first(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class to_channels_last(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

def build_norm_layer(dim, norm_layer, in_format='channels_last', out_format='channels_last', eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last': layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last': layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first': layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first': layers.append(to_channels_first())
    return nn.Sequential(*layers)

def build_act_layer(act_layer):
    if act_layer == 'ReLU': return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU': return nn.SiLU(inplace=True)
    elif act_layer == 'GELU': return nn.GELU()

class CenterFeatureScaleModule(nn.Module):
    def forward(self, query, center_feature_scale_proj_weight, center_feature_scale_proj_bias):
        return F.linear(query, weight=center_feature_scale_proj_weight, bias=center_feature_scale_proj_bias).sigmoid()

class Dynamic_Adaptive_Scan(nn.Module):
    def __init__(self, channels=64, kernel_size=1, dw_kernel_size=3, stride=1, pad=0, 
                 dilation=1, group=1, offset_scale=1.0, act_layer='GELU', norm_layer='LN', 
                 center_feature_scale=False, remove_center=False):
        super().__init__()
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels),
            build_norm_layer(channels, norm_layer, 'channels_first', 'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(channels, group * (kernel_size * kernel_size - remove_center) * 2)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)

    def forward(self, input_first, input_last):
        # input_first: N, C, H, W
        # input_last: N, H, W, C
        N, _, H, W = input_first.shape
        x1 = self.dw_conv(input_first)
        x_proj = input_last
        offset = self.offset(x1)
        mask_dcn = torch.ones(N, H, W, self.group, device=input_last.device, dtype=input_last.dtype)
        
        if DCNv3Function is not None:
            x = DCNv3Function.apply(
                input_last, offset, mask_dcn,
                self.kernel_size, self.kernel_size,
                self.stride, self.stride,
                self.pad, self.pad,
                self.dilation, self.dilation,
                self.group, self.group_channels,
                self.offset_scale,
                256,
                self.remove_center)
        else:
            # Fallback if DCN isn't available (should not happen in proper environment)
            x = input_last
            
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# =========================================================================
# Combined Core (VAMamba + DAMamba + SEM-Net base)
# =========================================================================

class CombinedAdaptiveMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        # Mamba Core
        self.mamba = Mamba(
            d_model=dim, 
            d_state=d_state,  
            d_conv=d_conv,    
            expand=expand,    
        )
        
        # VAMamba Component
        self.cnn_score_map = CNNScoreMap(
            in_channels=self.dim, 
            embed_dim=self.dim, 
        )
        
        # DAMamba Component
        # number of heads matching typically d_model/head_dim, assuming head_dim 16 or dim//2
        num_group = dim // 16 if dim >= 16 else 1
        self.da_scan = Dynamic_Adaptive_Scan(channels=self.dim, group=num_group)

    def forward(self, x, pe, mask=None, return_path=False):
        # x: [B, C, H, W]
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            
        B, C, H, W = x.shape
        assert C == self.dim
        n_tokens = H * W
        
        # ----------------------------------------------------
        # 1. DAMamba: Dynamic Adaptive Scan (Micro-Path)
        # ----------------------------------------------------
        # The DAMamba shifts the features to wrap around objects locally 
        # using Deformable Convolutions *before* passing into SSM sequence.
        # da_scan expects (N, C, H, W) and (N, H, W, C)
        x_last = x.permute(0, 2, 3, 1).contiguous()
        x_adapted = self.da_scan(x, x_last) # Output is [B, C, H, W]
        
        # ----------------------------------------------------
        # 2. VAMamba: Structure-Aware Start & Macro-Path
        # ----------------------------------------------------
        orig_score_map = self.cnn_score_map(x_adapted) # [B, H, W]
        
        # Dynamically set patch size: Attempt 4x4, fallback to 2x2 or 1x1 if H/W not divisible
        patch_size = 1
        for ps in [4, 2]:
            if H % ps == 0 and W % ps == 0:
                patch_size = ps
                break
                
        if patch_size > 1:
            # Pool scores to patch level
            score_map = F.avg_pool2d(orig_score_map.unsqueeze(1), kernel_size=patch_size, stride=patch_size).squeeze(1)
        else:
            score_map = orig_score_map
            
        # GPU Vectorized Traversal (Importance-driven dynamic path)
        B, H_p, W_p = score_map.shape
        num_patches = H_p * W_p
        
        # Flatten and sort the score map to determine the structural priority order
        flat_scores = score_map.view(B, num_patches)
        sorted_patch_indices = torch.argsort(flat_scores, dim=1, descending=True) # [B, num_patches]
        
        # ----------------------------------------------------
        # 3. Sequencing and Mamba Forward
        # ----------------------------------------------------
        x_flat = x_adapted.view(B, C, H, W) 
        
        if patch_size == 1:
            order_tensor = sorted_patch_indices # [B, n_tokens]
            pe_reordered = torch.gather(pe.unsqueeze(0).expand(B, -1, -1), 1, order_tensor.unsqueeze(-1).expand(-1, -1, pe.shape[-1]))
            x_b = x_adapted.view(B, C, n_tokens)
            x_reordered = torch.gather(x_b, 2, order_tensor.unsqueeze(1).expand(-1, C, -1))
        else:
            # Map patch ordering up to pixel ordering in a vectorized way
            p_y = sorted_patch_indices // W_p
            p_x = sorted_patch_indices % W_p
            
            # create pixel offsets inside the patch
            offsets_y = torch.arange(patch_size, device=x.device).view(patch_size, 1).repeat(1, patch_size).flatten()
            offsets_x = torch.arange(patch_size, device=x.device).view(1, patch_size).repeat(patch_size, 1).flatten()
            
            # broadcast to compute all pixel indices
            pixel_y = (p_y.unsqueeze(2) * patch_size) + offsets_y.view(1, 1, -1)
            pixel_x = (p_x.unsqueeze(2) * patch_size) + offsets_x.view(1, 1, -1)
            
            order_tensor = (pixel_y * W + pixel_x).view(B, n_tokens) # [B, n_tokens]
            
            # Gather pe and x intelligently across batch via gather
            pe_expanded = pe.unsqueeze(0).expand(B, -1, -1)
            pe_reordered = torch.gather(pe_expanded, 1, order_tensor.unsqueeze(2).expand(-1, -1, self.dim))
            
            x_b = x_adapted.view(B, C, n_tokens)
            x_reordered = torch.gather(x_b, 2, order_tensor.unsqueeze(1).expand(-1, C, -1))

        # Add PE, transpose to [B, L, C] for Mamba
        x1_flat = x_reordered.transpose(1, 2) + pe_reordered
        x1_norm = self.norm(x1_flat)
        x1_mamba = self.mamba(x1_norm) # [B, n_tokens, C]
        
        # Un-shuffle the output to [B, C, H, W]
        inverse_order = torch.argsort(order_tensor, dim=1) # The inverse mapping!
        
        x1_mamba_t = x1_mamba.transpose(1, 2) # [B, C, n_tokens]
        out_flat = torch.gather(x1_mamba_t, 2, inverse_order.unsqueeze(1).expand(-1, C, -1))
        out = out_flat.view(B, C, H, W)
        
        # Stash for validation hook visualization
        self.last_scan_orders = sorted_patch_indices
        self.last_patch_size = patch_size
        self.last_W_p = W_p
            
        if return_path:
            return out, sorted_patch_indices
        return out
