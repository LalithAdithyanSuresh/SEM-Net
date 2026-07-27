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
# Inward Spiral / Distance-Based Boundary-to-Center Pathing Helper
# =========================================================================

def compute_inward_hole_distance(mask):
    """
    mask: [B, 1, H, W] tensor (1.0 = hole to be filled, 0.0 = known background).
    Returns: [B, H, W] distance levels (1 = outer hole boundary, 2 = 2nd ring, ..., N = hole center).
    """
    B, _, H, W = mask.shape
    dist = torch.zeros((B, H, W), device=mask.device, dtype=torch.float32)
    curr_mask = (mask > 0.5).float()
    
    # 3x3 maxpool acts as binary dilation of known background into hole
    pool = nn.MaxPool2d(3, stride=1, padding=1)
    
    max_iters = max(H, W) // 2 + 1
    for level in range(1, max_iters + 1):
        if curr_mask.max() == 0:
            break
        # Dilate known background (1 - curr_mask)
        dilated_known = pool(1.0 - curr_mask)
        # Boundary pixels are hole pixels adjacent to known region
        boundary = (curr_mask > 0.5) & (dilated_known > 0.5)
        dist = torch.where(boundary.squeeze(1), float(level), dist)
        # Remove identified boundary pixels for next inward ring level
        curr_mask = torch.where(boundary, 0.0, curr_mask)

    return dist

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
        
        # Combine learned structural saliency with local feature variance
        final_scores = scores + content_score
        return final_scores


class CombinedAdaptiveMambaLayer(nn.Module):
    """
    CAMINO-Fill Hybrid Mamba Layer:
    Combines Inward Spiral Boundary-to-Center Sequence Pathing (VAMamba)
    and Deformable Convolutional Micro-Warping (DAMamba) into a unidirectional Mamba SSM.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_groups=4):
        super().__init__()
        self.d_model = d_model
        
        # 1. Structural Score Generator (VAMamba)
        self.score_generator = CNNScoreMap(in_channels=d_model, embed_dim=d_model)
        
        # 2. Deformable Micro-Offset Predictor (DAMamba / DCNv3 Component)
        self.num_groups = num_groups
        self.offset_dim = num_groups * 2 * 3 * 3  # (x, y) offsets per sampling point per group
        self.mask_dim = num_groups * 3 * 3        # Modulation weights per sampling point per group
        
        self.offset_mask_head = nn.Conv2d(
            d_model, 
            self.offset_dim + self.mask_dim, 
            kernel_size=3, 
            padding=1
        )
        # Initialize offsets to zero and masks to ~0.5 (sigmoid 0)
        constant_(self.offset_mask_head.weight, 0.0)
        constant_(self.offset_mask_head.bias, 0.0)

        # 3. Core SSM Model (Mamba)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # 4. Output Fusion Gate (1x1 Conv for 4D Feature Maps)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
            nn.Sigmoid()
        )

    def _apply_deformable_sampling(self, x):
        """
        Calculates local offsets and modulation masks to spatially warp features 
        along semantic boundaries before sequence processing.
        """
        B, C, H, W = x.shape
        out = self.offset_mask_head(x)
        offsets = out[:, :self.offset_dim, :, :]
        masks = torch.sigmoid(out[:, self.offset_dim:, :, :])

        if DCNv3Function is not None and x.is_cuda:
            try:
                x_perm = x.permute(0, 2, 3, 1).contiguous() # [B, H, W, C]
                offsets_perm = offsets.permute(0, 2, 3, 1).contiguous()
                masks_perm = masks.permute(0, 2, 3, 1).contiguous()
                
                x_deformed = DCNv3Function.apply(
                    x_perm, 
                    offsets_perm, 
                    masks_perm,
                    3, 3, 1, 1, 1, 1, 1, 1, # kernel, stride, pad, dilation
                    self.num_groups, 
                    C // self.num_groups, 
                    1.0, 256
                )
                return x_deformed.permute(0, 3, 1, 2).contiguous()
            except Exception:
                pass

        # Pure PyTorch fallback for deformable sampling / modulation
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1) # [B, H, W, 2]
        
        avg_offset = offsets.view(B, self.num_groups, 2, 9, H, W).mean(dim=(1, 3)) # [B, 2, H, W]
        avg_offset = avg_offset.permute(0, 2, 3, 1) # [B, H, W, 2]
        
        norm_offset = torch.zeros_like(avg_offset)
        norm_offset[..., 0] = avg_offset[..., 0] / max(W - 1, 1)
        norm_offset[..., 1] = avg_offset[..., 1] / max(H - 1, 1)
        
        sampled_grid = base_grid + norm_offset
        x_warped = F.grid_sample(x, sampled_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        
        avg_mask = masks.view(B, self.num_groups, 9, H, W).mean(dim=(1, 2)).unsqueeze(1) # [B, 1, H, W]
        return x_warped * avg_mask

    def forward(self, x, pos=None, mask=None):
        """
        x: [B, C, H, W] tensor
        pos: optional positional encoding tensor
        mask: optional binary mask tensor [B, 1, H, W] (1.0 = hole, 0.0 = known)
        Returns: [B, C, H, W] tensor processed by Combined Adaptive Mamba
        """
        B, C, H, W = x.shape
        L = H * W
        
        # Step 1: Deformable Local Micro-Warping (DAMamba)
        x_deformed = self._apply_deformable_sampling(x)
        
        # Step 2: Saliency Score Map Calculation
        saliency_scores = self.score_generator(x_deformed) # [B, H, W]

        # Step 3: Inward Spiral Boundary-to-Center Pathing Order
        if mask is not None:
            if mask.dim() == 4 and mask.shape[1] == 1:
                m_tensor = mask
            elif mask.dim() == 4:
                m_tensor = mask[:, 0:1, :, :]
            else:
                m_tensor = mask.unsqueeze(1)

            if m_tensor.shape[-2:] != (H, W):
                m_tensor = F.interpolate(m_tensor.float(), size=(H, W), mode='nearest')

            # Compute inward distance levels (1 = outer hole boundary, 2 = 2nd ring, ..., N = center)
            inward_dist = compute_inward_hole_distance(m_tensor) # [B, H, W]
            m_flat = m_tensor.squeeze(1)

            # Known valid pixels (mask == 0) get top priority (+1000.0)
            # Hole pixels (mask == 1) are sorted inward from boundary (-inward_dist * 10.0)
            scores = (1.0 - m_flat.float()) * 1000.0 - (inward_dist * 10.0) + saliency_scores
        else:
            scores = saliency_scores
        
        scores_flat = scores.view(B, L) # [B, L]
        
        # Sort scores in descending order:
        # Order: 1. Known Background -> 2. Outer Hole Boundary -> 3. Inner Hole Rings -> 4. Center Hole
        sort_indices = torch.argsort(scores_flat, dim=-1, descending=True) # [B, L]
        
        # Save scan orders for visualization panels if requested
        self.last_scan_orders = sort_indices
        self.last_patch_size = 1
        self.last_W_p = W

        # Construct inverse permutation mapping to reconstruct 2D spatial grid later
        inverse_indices = torch.empty_like(sort_indices)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(B, L)
        inverse_indices[batch_idx, sort_indices] = torch.arange(L, device=x.device).expand(B, L)

        # Step 4: Sequence Flattening and Inward Spiral Permutation
        x_flat = x_deformed.view(B, C, L).permute(0, 2, 1) # [B, L, C]
        
        # Reorder 1D sequence tokens based on inward spiral adaptive score path
        x_permuted = torch.gather(
            x_flat, 
            dim=1, 
            index=sort_indices.unsqueeze(-1).expand(B, L, C)
        ) # [B, L, C]

        # Step 5: Core State-Space Model Processing (Mamba SSM)
        x_ssm = self.mamba(x_permuted) # [B, L, C]

        # Step 6: Inverse Permutation & Spatial Restoration
        x_restored_flat = torch.gather(
            x_ssm, 
            dim=1, 
            index=inverse_indices.unsqueeze(-1).expand(B, L, C)
        ) # [B, L, C]
        
        x_restored_2d = x_restored_flat.permute(0, 2, 1).view(B, C, H, W) # [B, C, H, W]

        # Step 7: Dynamic Output Fusion (Residual + Deformed + SSM)
        gate = self.fusion_gate(torch.cat([x, x_restored_2d], dim=1))
        out = gate * x_restored_2d + (1 - gate) * x

        return out
