import os
os.environ['TORCH_HOME'] = './torch_cache'
os.environ['MPLCONFIGDIR'] = './matplotlib_cache'

import argparse
import time
import torch
from src.config import Config
from src.models import InpaintingModel

def main():
    parser = argparse.ArgumentParser(description="Test SEM-Net Inference Time")
    parser.add_argument('--path', type=str, default='./PlacesTraining', help='Path to the model directory containing config.yml and checkpoints')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution of the input image')
    parser.add_argument('--runs', type=int, default=100, help='Number of iterations to average the timing')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    args = parser.parse_args()

    print(f"[*] Setting up configuration for {args.path}...")
    config_path = os.path.join(args.path, 'config.yml')
    if not os.path.exists(config_path):
        print(f"Error: {config_path} does not exist.")
        return

    config = Config(config_path)
    config.PATH = args.path
    config.MODE = 2
    config.MODEL = 2
    config.WORLD_SIZE = 1
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Loading model onto {config.DEVICE}...")
    model = InpaintingModel(config).to(config.DEVICE)
    model.load()
    model.eval()

    print(f"[*] Creating dummy input of size {args.resolution}x{args.resolution}...")
    dummy_image = torch.randn(1, 3, args.resolution, args.resolution).to(config.DEVICE)
    dummy_mask = torch.randn(1, 1, args.resolution, args.resolution).to(config.DEVICE)

    # Warmup
    print(f"[*] Running {args.warmup} warmup iterations...")
    torch.backends.cudnn.benchmark = True # Enables highly optimized convolution kernels
    
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy_image, dummy_mask)

    # Timing
    print(f"[*] Running {args.runs} timed iterations (with cuDNN Benchmark)...")
    times = []
    with torch.no_grad():
        for _ in range(args.runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            
            _ = model(dummy_image, dummy_mask)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0

    print("=========================================")
    print(f"Inference Time Summary ({args.resolution}x{args.resolution}):")
    print(f"Average Time: {avg_time:.4f} seconds")
    print(f"FPS:          {fps:.2f} frames/sec")
    print("=========================================")

if __name__ == '__main__':
    main()
