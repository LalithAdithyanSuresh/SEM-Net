# Optimizing Adaptive Fused Mamba

## Background and Analysis

Currently, the `CombinedAdaptiveMambaLayer` integrates DAMamba (micro-level tracking via DCNv3) and VAMamba (macro-level scanning via ViT) with standard 1D Mamba processing. While this provides highly dynamic structure-aware processing, there are significant bottlenecks and parameter overheads:

1. **Parameter Heavy Macro-Path:** `ViTScoreMap` uses a full `TransformerEncoderLayer` at every block to generate an attention map. Transformers run in $O(N^2)$ and increase the parameter count linearly per Mamba layer.
2. **Computational Bottleneck (Python Loop):** The `adaptive_patch_traversal` function uses a standard Python `while` loop to calculate sorting indices for patches. This operations runs on the CPU, breaks the GPU compute graph, prevents batch vectorization, and is exceptionally slow.
3. **Inefficient Sequencing:** Rearranging patches to 1D and back using dynamically computed CPU indices prevents parallel processing inside the Mamba core and introduces latency.

**EAMamba's Approach for Efficiency:**
EAMamba (Efficient-Adaptive Mamba) maintains high performance while staying efficient by:
- Using **pre-computed scanning paths** (Hilbert, Z-order, ZigZag) rather than dynamically calculating a path sequentially. It vectors these transforms instantly in PyTorch (`ScanTransform`).
- Utilizing lightweight Channel Attention (CCA) and gated conv-feed forwards (GDFN) instead of heavy Transformers to introduce spatial and cross-channel awareness.
- A modified `ExtendedMamba` core that natively maps different scan directions without manually reshaping tensors prior to a standard `mamba_ssm` invocation.

---

## Proposed Changes

To make your Adaptive Fusion efficient while retaining its core functionality, we propose the following hybrid approach:

### 1. Removing the Python Traversal Loop
Instead of searching for the highest score dynamically pixel-by-pixel (which causes the CPU bottleneck), we can vectorize the macro pathing on the GPU.
*   **Option A (Fast Sorting):** Compute the `score_map` per patch exactly as you do now, but use `torch.argsort()` on the flattened scores to determine the sequential order instantly without a `while` loop.
*   **Option B (EAMamba Hybrid):** Use EAMamba's pre-computed sweeps (e.g., ZigZag, Z-order). Instead of creating a custom 1D path, the module runs Mamba over 2 or 4 efficient static paths and uses your `score_map` to **adaptively gate/weight** which paths contribute the most at each spatial region. This is massively faster and better matches "Efficient" architectures.

### 2. Parameter Reduction in `ViTScoreMap`
Replace the $O(N^2)$ `TransformerEncoder` within `ViTScoreMap` with a high-efficiency layout.
*   We can use EAMamba's `CCABlock` (Custom Channel Attention) or a lightweight sequence of depthwise convolutions to generate the structure and variance scores. This will reduce parameters by >60% in that module while retaining spatial awareness.

### 3. Integrating `ExtendedMamba`
Replace standard `mamba_ssm.Mamba` with EAMamba's `ExtendedMamba`. 
*   `ExtendedMamba` natively handles spatial merging and 2D-to-1D mapping inside the scan, allowing us to drop your custom un-shuffling code and significantly reducing forward-pass memory allocation. 

---

## Open Questions

> [!IMPORTANT]
> Please review and answer the following before we proceed:
 
1. **Dynamic Patch Order vs. Adaptive Multi-Path:** Do you strictly want the model to generate a *custom order* of patches (e.g., sorting by the highest score), or are you open to **Option B** where we use fast, fixed paths (like Hilbert/Z-order) but *adaptively gate* them using your attention mechanism (similar to EAMamba)? Option B is far more parameter and compute efficient.
2. **Patch Size (8x8):** You mentioned the 8x8 patch size is currently used. If we optimize the pathing to run entirely vectorized on the GPU, we could potentially lower this to 4x4 or even operate densely depending on resolution. Would you like to keep the 8x8 clustering, or should we make it resolution-adaptive?
3. **DCNv3 Retention:** Do you want to keep the DAMamba (DCNv3) micro-path exactly as it is, or should we also look at simplifying its deformable convolutions to save parameters?

## Verification Plan

### Automated Tests
*   **Runtime Profiling:** Profile the forward pass time per batch to guarantee generation time drops significantly from the current average (~1.8-2.0 seconds).
*   **Train Run:** Execute `recalculate_metrics.py` or a 1-epoch `train.py` run to ensure gradients flow correctly back through the new vectorized pathing.
*   **Parameter Count:** Log the exact parameter count before and after the modification to prove reduction.
