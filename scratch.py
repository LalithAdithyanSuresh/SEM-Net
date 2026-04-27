import torch
from src.dataset import Dataset
from config import Config
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F

mask = (np.ones((256, 256)) * 255).astype(np.uint8)
img = Image.fromarray(mask)
img_t = F.to_tensor(img).float()
print("Max value:", img_t.max(), "Min value:", img_t.min())
print("Dtype:", img_t.dtype)
