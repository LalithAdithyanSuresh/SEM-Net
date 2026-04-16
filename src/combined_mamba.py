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

class ViTScoreMap(nn.Module):
    def __init__(self, in_channels, patch_size=8, embed_dim=64, num_layers=1, num_heads=2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size)**2, embed_dim)) 
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != self.patch_size and x.shape[1] != self.embed_dim:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, 'H, W must be divisible by patch_size'
        
        x_unfold = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = x_unfold.transpose(1, 2)  
        
        if patches.shape[-1] != self.embed_dim:
            patches = nn.Linear(patches.shape[-1], self.embed_dim, device=x.device)(patches)
        
        N_patch = patches.shape[1]
        
        if self.pos_embed.shape[1] != N_patch:
            orig_N = self.pos_embed.shape[1]
            orig_size = int(orig_N ** 0.5)
            pos_embed = F.interpolate(
                self.pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2),
                size=(H // self.patch_size, W // self.patch_size),
                mode='bilinear'
            ).permute(0, 2, 3, 1).flatten(1, 2)
        else:
            pos_embed = self.pos_embed

        patches = patches + 0.1 * pos_embed
        feats = self.transformer(patches)
        scores = self.score_head(feats).squeeze(-1)
        scores = torch.sigmoid(scores)
        
        patch_variance = torch.var(patches, dim=-1)  
        content_score = torch.sigmoid(patch_variance * 10)  
        
        final_scores = 0.7 * scores + 0.3 * content_score
        
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        score_map = final_scores.view(B, H_patch, W_patch)
        return score_map

def adaptive_patch_traversal(score_map):
    H, W = score_map.shape
    device = score_map.device
    visited = torch.zeros_like(score_map, dtype=torch.bool, device=device)
    order = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while len(order) < H * W:
        if len(order) == 0:
            start_idx = torch.argmax(score_map).item()
            start_i, start_j = divmod(start_idx, W)
        else:
            unvisited_scores = score_map[~visited]
            if unvisited_scores.numel() == 0:
                break
            max_unvisited_score_idx = torch.argmax(unvisited_scores).item()
            unvisited_indices = (~visited).nonzero(as_tuple=False)
            start_i, start_j = unvisited_indices[max_unvisited_score_idx]
            start_i, start_j = start_i.item(), start_j.item()
        
        current_i, current_j = start_i, start_j
        visited[current_i, current_j] = True
        order.append((current_i, current_j))
        
        while True:
            best_neighbor = None
            best_score = float('-inf')
            
            for di, dj in directions:
                ni, nj = current_i + di, current_j + dj
                if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj]:
                    if score_map[ni, nj] > best_score:
                        best_score = score_map[ni, nj]
                        best_neighbor = (ni, nj)
            
            if best_neighbor is not None:
                current_i, current_j = best_neighbor
                visited[current_i, current_j] = True
                order.append((current_i, current_j))
            else:
                break
    
    return order


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
        patch_size_div = 8 # Default patch divisor based on features
        self.vit_score_map = ViTScoreMap(
            in_channels=self.dim, 
            patch_size=8,
            embed_dim=self.dim, 
            num_layers=1, 
            num_heads=2
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
        # We compute the score map for the feature.
        # We ensure patch_size divides H and W, if not fallback to global
        patch_size = 8
        if H % patch_size != 0: patch_size = 1
        
        if patch_size == 1:
            # Bypass score map if dimensions are too small or prime
            score_map = torch.ones(B, H, W, device=x.device)
            scan_orders = [[(i, j) for i in range(H) for j in range(W)]] * B
        else:
            self.vit_score_map.patch_size = patch_size
            score_map = self.vit_score_map(x_adapted)
            scan_orders = []
            for b in range(B):
                order = adaptive_patch_traversal(score_map[b])
                scan_orders.append(order)
                
        # ----------------------------------------------------
        # 3. Sequencing and Mamba Forward
        # ----------------------------------------------------
        
        # Flatten based on macro path order
        H_p, W_p = (H//patch_size), (W//patch_size)
        x_reordered = torch.zeros_like(x_adapted.view(B,C,n_tokens))
        pe_reordered = torch.zeros(B, n_tokens, self.dim, device=x.device)
        
        for b in range(B):
            if patch_size > 1:
                # Map patch-level order back to pixel-level order
                order_indices = []
                for p_i, p_j in scan_orders[b]:
                    # For each patch, add all pixels within it sequentially
                    for r in range(patch_size):
                        for c in range(patch_size):
                            idx = (p_i * patch_size + r) * W + (p_j * patch_size + c)
                            order_indices.append(idx)
                order_tensor = torch.tensor(order_indices, device=x.device, dtype=torch.long)
            else:
                order_tensor = torch.arange(n_tokens, device=x.device, dtype=torch.long)
                
            x_b = x_adapted[b].view(C, n_tokens)
            x_reordered[b] = x_b[:, order_tensor]
            # applying permuted PE
            pe_reordered[b] = pe[order_tensor, :]
            
        # Add PE, transpose to [B, L, C] for Mamba
        x1_flat = x_reordered.transpose(-1, -2) + pe_reordered
        x1_norm = self.norm(x1_flat)
        x1_mamba = self.mamba(x1_norm)
        
        # Un-shuffle the output to [B, C, H, W]
        out = torch.zeros(B, C, n_tokens, device=x.device)
        x1_mamba = x1_mamba.transpose(-1, -2)
        for b in range(B):
            if patch_size > 1:
                order_indices = []
                for p_i, p_j in scan_orders[b]:
                    for r in range(patch_size):
                        for c in range(patch_size):
                            idx = (p_i * patch_size + r) * W + (p_j * patch_size + c)
                            order_indices.append(idx)
                order_tensor = torch.tensor(order_indices, device=x.device, dtype=torch.long)
            else:
                order_tensor = torch.arange(n_tokens, device=x.device, dtype=torch.long)
                
            inverse_order = torch.empty_like(order_tensor)
            inverse_order[order_tensor] = torch.arange(n_tokens, device=x.device)
            out[b] = x1_mamba[b, :, inverse_order]

        out = out.reshape(B, C, H, W)
        
        # Stash for validation hook visualization
        if patch_size > 1:
            self.last_scan_orders = scan_orders
        else:
            self.last_scan_orders = None
            
        if return_path:
            return out, scan_orders
        return out
