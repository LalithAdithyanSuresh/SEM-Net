import torch
import os

path = '/home/snuc/Desktop/SEM-NETHybrid/SEM-Net/updated_spiral_8x8/InpaintingModel_gen.pth'
try:
    data = torch.load(path)
    print("Successfully loaded!")
    print(f"Iteration: {data.get('iteration', 'N/A')}")
except Exception as e:
    print(f"Failed to load: {e}")
