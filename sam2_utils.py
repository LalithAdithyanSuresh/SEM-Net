# sam2_utils.py
"""Utility helpers for loading and running inference with the SAM2 model.

This module isolates SAM2‑specific imports so that the main script can optionally
use SAM2 without importing its heavy dependencies when the user only wants the
original SAM / FastSAM functionality.

Requirements:
- Install the Segment Anything 2 package (e.g.)
  ```
  pip install git+https://github.com/facebookresearch/segment-anything.git
  ```
- Have a SAM2 checkpoint YAML config file (e.g., `sam2_hiera_l.yaml`) and the
  corresponding `.pt` weight file. The weight file path is typically inferred by
  the SAM2 repository based on the YAML location.

The functions below provide a thin wrapper that returns binary masks compatible
with the existing mask‑generation pipeline.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Optional


def load_sam2_model(config_path: str, device: str = "cuda"):
    """Load a SAM2 model based on a YAML configuration file.

    Args:
        config_path: Path to the SAM2 YAML config (e.g., `sam2_hiera_l.yaml`).
        device: Torch device string (e.g., "cuda" or "cpu").

    Returns:
        An instantiated ``SAM2ImagePredictor`` ready for inference.
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise ImportError(
            "SAM2 libraries are not installed. Install them with\n"
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    # Determine checkpoint path.
    ckpt_path: Optional[str] = None
    if config_path.lower().endswith('.pt'):
        ckpt_path = config_path
        config_path = os.path.splitext(config_path)[0] + ".yaml"
    else:
        possible_pt = os.path.splitext(config_path)[0] + ".pt"
        if os.path.isfile(possible_pt):
            ckpt_path = possible_pt
        else:
            raise FileNotFoundError(
                f"Could not locate SAM2 checkpoint for config '{config_path}'."
            )

    sam2_model = build_sam2(config_path, ckpt_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def run_sam2_inference(predictor, image_path: str, device: str = "cuda") -> List[np.ndarray]:
    """Run SAM2 inference on a single image and return binary masks.

    Args:
        predictor: ``SAM2ImagePredictor`` instance returned by ``load_sam2_model``.
        image_path: Path to the input image.
        device: Device string – kept for API compatibility.

    Returns:
        List of binary ``np.ndarray`` masks (bool arrays).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    pred = predictor.predict(point_coords=None, point_labels=None, box=None, multimask_output=False)
    masks_tensor = pred.get("masks")
    if masks_tensor is None:
        return []
    masks_np = masks_tensor.cpu().numpy()
    return [(m > 0.5) for m in masks_np]
