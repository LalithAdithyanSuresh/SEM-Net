"""
benchmark.py — Parameter count + inference time for SEM-Net (updated_spiral)
Run with:
    conda activate inpaint_env_3.10
    cd /home/snuc/Desktop/SEM-NETHybrid/SEM-Net
    python benchmark.py --path ./updated_spiral_8x8
"""
import os, time, argparse
import torch
import numpy as np
from src.config import Config
from src.models import InpaintingModel

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./updated_spiral_8x8')
parser.add_argument('--size', type=int, default=256,  help='Input image size')
parser.add_argument('--runs', type=int, default=100,  help='Warmup+timed forward passes')
parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations (excluded from timing)')
args = parser.parse_args()

# ── Config ──────────────────────────────────────────────────────────────────
config = Config(os.path.join(args.path, 'config.yml'))
config.PATH   = args.path
config.MODE   = 2          # inference
config.MODEL  = 2
config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice : {config.DEVICE}")
print(f"Image  : {args.size}×{args.size}")

# ── Load model ───────────────────────────────────────────────────────────────
model = InpaintingModel(config).to(config.DEVICE)
model.load()
model.eval()

# ── Parameter count ──────────────────────────────────────────────────────────
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Generator only (discriminator not used at inference)
gen = model.generator
if hasattr(gen, 'module'):
    gen = gen.module
gen_params     = sum(p.numel() for p in gen.parameters())
gen_trainable  = sum(p.numel() for p in gen.parameters() if p.requires_grad)

print(f"\n{'='*55}")
print(f"  Parameter Count")
print(f"{'='*55}")
print(f"  Generator          : {gen_params:>15,}  ({gen_params/1e6:.2f} M)")
print(f"  Generator (train.) : {gen_trainable:>15,}  ({gen_trainable/1e6:.2f} M)")
print(f"  Full model (G+D)   : {total_params:>15,}  ({total_params/1e6:.2f} M)")

# ── Dummy input ───────────────────────────────────────────────────────────────
H = W = args.size
dummy_img  = torch.rand(1, 3, H, W, device=config.DEVICE)
dummy_mask = (torch.rand(1, 1, H, W, device=config.DEVICE) > 0.6).float()

# Precompute scaled masks (same as sem.py / models.py forward)
import torch.nn.functional as F
m_half    = F.interpolate(dummy_mask, size=[H//2,  W//2],  mode='nearest')
m_quarter = F.interpolate(dummy_mask, size=[H//4,  W//4],  mode='nearest')
m_tiny    = F.interpolate(dummy_mask, size=[H//8,  W//8],  mode='nearest')

# ── Inference timing ─────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Inference Timing  (warmup={args.warmup}, timed runs={args.runs - args.warmup})")
print(f"{'='*55}")

latencies = []
with torch.no_grad():
    for i in range(args.runs):
        if config.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = model(dummy_img, dummy_mask)

        if config.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= args.warmup:
            latencies.append((t1 - t0) * 1000)  # ms

latencies = np.array(latencies)
print(f"  Mean  latency  : {latencies.mean():.2f} ms / image")
print(f"  Median latency : {np.median(latencies):.2f} ms / image")
print(f"  Std            : {latencies.std():.2f} ms")
print(f"  Min / Max      : {latencies.min():.2f} / {latencies.max():.2f} ms")
print(f"  Throughput     : {1000/latencies.mean():.2f} images/sec")

if config.DEVICE.type == 'cuda':
    mem_mb = torch.cuda.max_memory_allocated(config.DEVICE) / 1024**2
    print(f"  Peak GPU mem   : {mem_mb:.1f} MB")

print(f"{'='*55}\n")
