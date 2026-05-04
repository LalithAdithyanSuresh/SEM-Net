# SEM-Net Hybrid Architecture: Known-to-Unknown Spiral Scanning Implementation Plan

## 1. Goal and Context
The objective is to upgrade the Mamba sequence processing within SEM-Net (specifically `CombinedAdaptiveMambaLayer` in `src/combined_mamba.py`) to incorporate a **Known-to-Unknown Spiral Path**.

### Background Context
- **SEM-Net** is a U-Net style architecture (7 macro stages) where each stage contains `TransformerBlock`s.
- Each `TransformerBlock` calls `CombinedAdaptiveMambaLayer`, which combines VAMamba (Score-based pathing) and DAMamba (Deformable Convolutions).
- **Current Issue**: The 1D path generation inside `CombinedAdaptiveMambaLayer` ignores the mask. The Mamba hidden state ($h_t$) can drift/blur when scanning through long sequences of masked (unknown) regions before hitting valid features.
- **Masking Convention in Codebase**:
  - `mask == 0`: Valid Image / Known Region / Background.
  - `mask == 1`: Hole / Unknown Region / To-be-inpainted.

### The Solution
1. **Dilated CNN Score Map**: Replace standard 3x3 convolutions with Dilated Convolutions in `CNNScoreMap` to expand the receptive field.
2. **Distance Transform**: Compute how far every pixel inside the hole (`mask == 1`) is from the known boundary.
3. **Path Fusion Logic**: Combine the Distance Map, Mask constraints, and CNN scores into a single `final_scores` map.
   - Boost known regions (`mask == 0`) massively so they are processed first.
   - Boost hole regions (`mask == 1`) proportionally to their distance from the center, so outer edges are processed before the deep center (Spiral).
4. `argsort(final_scores, descending=True)` generates the sequence path.

---

## 2. Architecture Diagram (ASCII)

```text
====================================================================================
INITIAL INPUTS (Inside one TransformerBlock)
====================================================================================
      Input Features (x)                     Binary Mask
        [B, C, H, W]                         [B, 1, H, W]
             |                                    |
             v                                    v
====================================================================================
MICRO-DEFORMATION (DAMamba) & SPATIAL ANALYSIS
====================================================================================
  +-----------------------+              +-----------------------+
  | Dynamic Adaptive Scan |              |   Distance Transform  |
  |     (DA_Scan)         |              |  (Morphology/Approx)  |
  +-----------------------+              +-----------------------+
             |                                    |
  (Deformable Convolutions bend          (Calculates how far every
   features along structural edges)       masked pixel is from the
             |                                 valid boundary)
             |                                    |
             v                                    |
       [x_adapted]                                |
       /         \                                |
      /           \                               |
     v             v                              v
+----------+  +-------------------+        [Distance Map]
| Sequence |  | Dilated CNN Score |               |
| Reorder  |  |       Map         |               |
|  Block   |  +-------------------+               |
| (Wait)   |           |                          |
+----------+           v                          |
                [Base Scores]                     |
                       \                          /
                        \                        /
====================================================================================
PATH FUSION LOGIC (The "Spiral & Constraint" Engine)
====================================================================================
                         \                      /
                          v                    v
                  +-----------------------------------+
                  |        Path Fusion Logic          |
                  |-----------------------------------|
                  | 1. Known-to-Unknown Constraint:   |
                  |    Add massive score to Mask==0   | <--- (Known Region)
                  |                                   |
                  | 2. Spiral Distance Boost:         |
                  |    Add Distance Map to Mask==1    | <--- (Hole Region)
                  |                                   |
                  | 3. Local Texture/Edge Prior:      |
                  |    Add Dilated CNN Base Scores    |
                  +-----------------------------------+
                                   |
                                   v
                             [Final Scores]
                                   |
                                   v
                  +-----------------------------------+
                  | torch.argsort(descending=True)    |
                  +-----------------------------------+
                                   |
                                   v
                            [Path Indices]
                                   |
====================================================================================
SEQUENCE MODELING (Mamba) & RECONSTRUCTION
====================================================================================
                                   |
                                   v
  +----------+             +-------------------+             +-------------------+
  | Sequence |             |                   |             | Inverse Reorder   |
  | Reorder  | ----------> | Mamba Core (SSM)  | ----------> | (Restore 2D grid  |
  |  Block   | (1D Stream) |  [B, n_tokens, C] | (1D Stream) |  from 1D stream)  |
  | (Flatten)|             |                   |             |                   |
  +----------+             +-------------------+             +-------------------+
                                                                       |
                                                                       v
                                                                Output Features
                                                                 [B, C, H, W]
                                                                       |
                                                                       v
                                                               +---------------+
                                                               | Residual Add  | <--- (From Original x)
                                                               +---------------+
```

---

## 3. Implementation Steps (Code Modifications)

### File to Modify: `src/combined_mamba.py`

#### Step A: Upgrade `CNNScoreMap` to use Dilated Convolutions
In `CNNScoreMap.__init__`:
- Modify `self.conv1` and `self.conv2` to include `dilation=2` and `padding=2` (or appropriate dilation rates) to expand the receptive field for better local texture priors.

#### Step B: Implement GPU-Accelerated Distance Transform
Inside `CombinedAdaptiveMambaLayer.forward(self, x, pe, mask=None, return_path=False)`:
- If `mask` is provided, we need to compute the distance transform.
- *PyTorch limitation*: PyTorch does not have a native `scipy.ndimage.distance_transform_edt`.
- *Solution*: Implement a fast approximation using morphological erosion via max pooling, or a custom distance map convolution loop running a few iterations on the GPU.

#### Step C: Implement Path Fusion Logic
Inside `CombinedAdaptiveMambaLayer.forward`:
- Extract `mask`. Downsample/pool the `mask` if `patch_size > 1` (similar to how `orig_score_map` is pooled).
- Compute `base_scores = orig_score_map` (from Dilated CNN).
- Compute `distance_map` for `mask == 1`.
- Construct `final_scores`:
  ```python
  # 1. Known pixels (mask == 0) get massive priority
  known_boost = (1.0 - mask_pooled) * 10000.0 
  
  # 2. Spiral boost inside hole (mask == 1). 
  # Higher distance from center (closer to edge) = higher score.
  # Assuming distance_map is normalized [0, 1] inside the hole
  spiral_boost = mask_pooled * distance_map * 100.0
  
  # 3. Base scores (Local texture guidance)
  texture_scores = base_scores * 1.0
  
  # Combine
  final_scores = known_boost + spiral_boost + texture_scores
  ```
- Generate path: `sorted_patch_indices = torch.argsort(final_scores.view(B, -1), dim=1, descending=True)`

#### Step D: Refactoring Validation Visualization
- Ensure `last_scan_orders` logic in `src/sem.py` visualization still functions smoothly with the new distance-based paths. The path plot should visibly start outside the mask and spiral in.

---

## 4. Verification Plan
1. **Sanity Check**: Run `python src/combined_mamba.py` or a dedicated test script to instantiate the layer and pass a mock tensor and mask through it. Verify output shapes.
2. **Visual Path Verification**: Run training/evaluation for 5 iterations (`max_iterations = 5`) and review the visualization plots in `results/inpaint/validation/`. The "VAMamba Path" panel MUST show the sequence starting in the known regions, wrapping around the hole, and spiraling into the center.
3. **PSNR/L1 Verification**: Evaluate the new model against a baseline checkpoint to verify improved boundary sharpness.
