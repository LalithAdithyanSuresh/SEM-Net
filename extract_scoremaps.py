#!/usr/bin/env python3
"""
extract_scoremaps.py  –  Standalone VA & DA Score-Map Visualiser
================================================================
Loads the SEM-Net model, runs inference on up to 100 test images, and saves
high-resolution score-map visualisations with per-patch numerical values.

Output per image (saved to <output>/):
  VA_scoremaps/<name>_VA.png : Original image | VA score map with values
  DA_scoremaps/<name>_DA.png : Original image | DA offset magnitude map with values

Usage (on training server):
  python extract_scoremaps.py --path ./PlacesTraining \
      --input <test_images_dir> --mask <mask_dir> \
      [--output ./scoremap_results] [--num-images 100]

Usage (local with checkpoint):
  python extract_scoremaps.py --path ./checkpoints \
      --input ./test_images --mask ./test_masks \
      --num-images 10 --cell-px 50
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

# ── Project imports ──────────────────────────────────────────────────────
from src.config import Config
from src.models import InpaintingModel
from src.dataset import Dataset


# =========================================================================
# Helpers
# =========================================================================

def get_attn_layer(generator):
    """Get the CombinedAdaptiveMambaLayer from encoder_level1[0]."""
    if hasattr(generator, 'module'):
        return generator.module.encoder_level1[0].attn
    return generator.encoder_level1[0].attn


def postprocess_tensor(img_tensor):
    """[0,1] float tensor [B,C,H,W] → uint8 numpy [H,W,3]."""
    img = img_tensor[0].clamp(0, 1) * 255.0
    img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return img


def _get_font(size):
    """Try to load a readable font; fall back to PIL default."""
    for path in [
        "arial.ttf", "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_score_grid_image(score_map_2d, cell_px=40, cmap_name='plasma',
                           title='Score Map'):
    """
    Render a high-res image of a 2D score map with numerical values in each cell.

    Args:
        score_map_2d : np.ndarray [H_p, W_p] of float scores
        cell_px      : pixel size of each cell in the output image
        cmap_name    : matplotlib colormap name
        title        : title string drawn at the top

    Returns:
        PIL.Image of the rendered score grid
    """
    import matplotlib.pyplot as plt

    H_p, W_p = score_map_2d.shape
    out_w = W_p * cell_px
    out_h = H_p * cell_px
    title_h = max(30, cell_px)  # height reserved for title bar

    canvas = np.zeros((out_h + title_h, out_w, 3), dtype=np.uint8)

    cmap = plt.cm.get_cmap(cmap_name)

    # Normalise scores to [0, 1] for colour mapping
    s_min, s_max = float(score_map_2d.min()), float(score_map_2d.max())
    if s_max - s_min > 1e-8:
        normed = (score_map_2d - s_min) / (s_max - s_min)
    else:
        normed = np.zeros_like(score_map_2d)

    # Fill colour cells
    for row in range(H_p):
        for col in range(W_p):
            t = normed[row, col]
            r, g, b, _ = cmap(t)
            color = (int(r * 255), int(g * 255), int(b * 255))

            y0 = title_h + row * cell_px
            x0 = col * cell_px
            y1 = y0 + cell_px
            x1 = x0 + cell_px

            canvas[y0:y1, x0:x1] = color

            # 1-px dark border for visual separation
            dark = tuple(max(c - 50, 0) for c in color)
            canvas[y0, x0:x1] = dark
            canvas[min(y1 - 1, canvas.shape[0]-1), x0:x1] = dark
            canvas[y0:y1, x0] = dark
            canvas[y0:y1, min(x1 - 1, canvas.shape[1]-1)] = dark

    # Convert to PIL for text rendering
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    # ── Draw numerical values ────────────────────────────────────────────
    value_font_size = max(7, cell_px // 4)
    value_font = _get_font(value_font_size)

    for row in range(H_p):
        for col in range(W_p):
            val = score_map_2d[row, col]
            text = f"{val:.2f}"

            y_center = title_h + row * cell_px + cell_px // 2
            x_center = col * cell_px + cell_px // 2

            # Contrasting text colour
            t = normed[row, col]
            text_color = (0, 0, 0) if t > 0.5 else (255, 255, 255)

            # Centre the text
            bbox = draw.textbbox((0, 0), text, font=value_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x_center - tw // 2, y_center - th // 2), text,
                      fill=text_color, font=value_font)

    # ── Draw title bar ───────────────────────────────────────────────────
    draw.rectangle([(0, 0), (out_w, title_h)], fill=(30, 30, 30))
    title_font = _get_font(max(14, cell_px // 3))
    # Add range info to title
    full_title = f"{title}  [min={s_min:.3f}  max={s_max:.3f}]"
    draw.text((10, title_h // 4), full_title, fill=(255, 255, 255), font=title_font)

    return pil_img


def stitch_original_and_scoremap(original_pil, scoremap_pil):
    """Horizontally stitch the original image (upscaled) next to the score map."""
    target_h = scoremap_pil.height
    scale = target_h / original_pil.height
    target_w = int(original_pil.width * scale)
    orig_resized = original_pil.resize((target_w, target_h), Image.LANCZOS)

    gap = 6
    total_w = target_w + gap + scoremap_pil.width
    total_h = target_h

    canvas = Image.new('RGB', (total_w, total_h), (20, 20, 20))
    canvas.paste(orig_resized, (0, 0))
    canvas.paste(scoremap_pil, (target_w + gap, 0))

    return canvas


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract VA & DA score maps from SEM-Net',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--path', type=str, default='./PlacesTraining',
                        help='Checkpoint directory (must contain config.yml + model weights)')
    parser.add_argument('--input', type=str, default=None,
                        help='Test images directory (overrides config)')
    parser.add_argument('--mask', type=str, default=None,
                        help='Mask directory (overrides config)')
    parser.add_argument('--output', type=str, default='./scoremap_results',
                        help='Output directory for score map images')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of test images to process (default: 100)')
    parser.add_argument('--cell-px', type=int, default=40,
                        help='Pixel size of each patch cell in output (default: 40). '
                             'Increase for better readability on very dense maps.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index (default: 0)')
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    config_path = os.path.join(args.path, 'config.yml')
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}")
        sys.exit(1)

    config = Config(config_path)
    config.PATH = args.path
    config.MODE = 2   # test mode
    config.MODEL = 2  # inpaint model
    config.BATCH_SIZE = 1
    config.GPU = [args.gpu]

    if args.input is not None:
        config.TEST_INPAINT_IMAGE_FLIST = args.input
    if args.mask is not None:
        config.TEST_MASK_FLIST = args.mask

    # ── Device ───────────────────────────────────────────────────────────
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        print(f"Using GPU: {args.gpu}")
    else:
        config.DEVICE = torch.device("cpu")
        print("Using CPU")

    config.RANK = 0
    config.WORLD_SIZE = 1

    # ── Seed ─────────────────────────────────────────────────────────────
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model from: {args.path}")
    model = InpaintingModel(config).to(config.DEVICE)
    model.load()
    model.eval()
    print(f"Model loaded (iteration {model.iteration:,})")

    # ── Load dataset ─────────────────────────────────────────────────────
    test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST,
                           config.TEST_MASK_FLIST, augment=False, training=False)
    n_total = len(test_dataset)
    n_images = min(args.num_images, n_total)
    print(f"Test dataset: {n_total} images total, processing first {n_images}")

    indices = list(range(n_images))
    subset = Subset(test_dataset, indices)
    loader = DataLoader(subset, batch_size=1, num_workers=0, shuffle=False, pin_memory=True)

    # ── Output directory ─────────────────────────────────────────────────
    output_dir = args.output
    va_dir = os.path.join(output_dir, 'VA_scoremaps')
    da_dir = os.path.join(output_dir, 'DA_scoremaps')
    os.makedirs(va_dir, exist_ok=True)
    os.makedirs(da_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    print(f"  VA → {va_dir}")
    print(f"  DA → {da_dir}")

    # ── Get attention layer handle ───────────────────────────────────────
    attn_layer = get_attn_layer(model.generator)

    # ── Inference loop ───────────────────────────────────────────────────
    print(f"\nProcessing {n_images} images...")
    cell_px = args.cell_px

    for i, (images, masks) in enumerate(loader):
        images = images.to(config.DEVICE)
        masks = masks.to(config.DEVICE)

        # Forward pass
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _ = model(images, masks)

        # ── File name ────────────────────────────────────────────────────
        orig_idx = indices[i]
        name_base = test_dataset.load_name(orig_idx)
        name_stem = os.path.splitext(name_base)[0]

        # ── Original image as PIL ────────────────────────────────────────
        orig_np = postprocess_tensor(images)
        orig_pil = Image.fromarray(orig_np)
        H, W = orig_np.shape[:2]

        # =================================================================
        # VA Score Map
        # =================================================================
        va_raw = getattr(attn_layer, 'last_va_score_map', None)
        patch_size = getattr(attn_layer, 'last_patch_size', 1)

        if va_raw is not None:
            va_full = va_raw[0].cpu().float().numpy()  # [H_feat, W_feat]

            # Pool to patch-level if patches > 1 pixel
            if patch_size > 1 and va_full.shape[0] % patch_size == 0:
                va_t = torch.from_numpy(va_full).unsqueeze(0).unsqueeze(0)
                va_pooled = F.avg_pool2d(va_t, kernel_size=patch_size,
                                         stride=patch_size).squeeze().numpy()
            else:
                va_pooled = va_full

            va_grid = build_score_grid_image(
                va_pooled, cell_px=cell_px, cmap_name='plasma',
                title=f'VA Score – {name_base}  (patch={patch_size}×{patch_size})'
            )
            va_final = stitch_original_and_scoremap(orig_pil, va_grid)
            va_path = os.path.join(va_dir, f'{name_stem}_VA.png')
            va_final.save(va_path, quality=95)
        else:
            print(f"  [{i+1}] WARNING: VA score map not available")

        # =================================================================
        # DA Score Map (offset magnitudes)
        # =================================================================
        da_offset = getattr(attn_layer.da_scan, 'last_offset_map', None)

        if da_offset is not None:
            off = da_offset[0].cpu().float().numpy()  # [H_feat, W_feat, G*2]
            G = off.shape[-1] // 2
            # Magnitude per spatial position, averaged over offset groups
            mag = np.sqrt(off[..., :G] ** 2 + off[..., G:] ** 2).mean(axis=-1)

            # Pool to patch level to match VA resolution
            if patch_size > 1 and mag.shape[0] % patch_size == 0:
                mag_t = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float()
                mag_pooled = F.avg_pool2d(mag_t, kernel_size=patch_size,
                                          stride=patch_size).squeeze().numpy()
            else:
                mag_pooled = mag

            da_grid = build_score_grid_image(
                mag_pooled, cell_px=cell_px, cmap_name='jet',
                title=f'DA Offset Mag – {name_base}  (patch={patch_size}×{patch_size})'
            )
            da_final = stitch_original_and_scoremap(orig_pil, da_grid)
            da_path = os.path.join(da_dir, f'{name_stem}_DA.png')
            da_final.save(da_path, quality=95)
        else:
            print(f"  [{i+1}] WARNING: DA offset map not available")

        if (i + 1) % 10 == 0 or (i + 1) == n_images:
            print(f"  [{i+1:3d}/{n_images}] processed")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Done! {n_images} score-map pairs saved.")
    print(f"  VA score maps → {os.path.abspath(va_dir)}")
    print(f"  DA score maps → {os.path.abspath(da_dir)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
