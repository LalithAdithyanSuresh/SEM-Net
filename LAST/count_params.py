import torch
from src.config import Config
from src.models import InpaintingModel
import os
import sys

# Suppress some outputs
sys.path.append(os.getcwd())

def main():
    config_path = './checkpoints/config.yml'
    if not os.path.exists(config_path):
        print("Config not found at ./checkpoints/config.yml")
        return

    config = Config(config_path)
    # Force CPU for counting to avoid OOM or CUDA issues
    config.DEVICE = torch.device('cpu')
    config.GPU = []

    print("Initializing model to count parameters (this may take a moment)...")
    try:
        model = InpaintingModel(config)
        
        def count_parameters(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        gen_params = count_parameters(model.generator)
        dis_params = count_parameters(model.discriminator)

        print("\n" + "="*30)
        print(f"Generator:     {gen_params:,} ({gen_params / 1e6:.2f}M)")
        print(f"Discriminator: {dis_params:,} ({dis_params / 1e6:.2f}M)")
        print("-" * 30)
        print(f"Total:         {gen_params + dis_params:,} ({(gen_params + dis_params) / 1e6:.2f}M)")
        print("="*30)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
