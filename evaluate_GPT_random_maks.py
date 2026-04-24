import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import Dataset, generate_stroke_mask
from src.models import InpaintingModel
from src.utils import create_dir
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import torchvision
from PIL import Image
from cleanfid import fid
import csv
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import io

# ---------------- VISUALIZATION HELPERS ---------------- #
def get_mamba_path_image(model, gt_pil):
    """Extracts scan order from model and draws it over the GT image."""
    try:
        # Hijack the last scan orders from encoder 1
        if hasattr(model.generator, 'module'):
            layer = model.generator.module.encoder_level1[0].attn
        else:
            layer = model.generator.encoder_level1[0].attn
            
        scan_tensor = getattr(layer, 'last_scan_orders', None)
        if scan_tensor is None:
            return gt_pil
            
        indices = scan_tensor[0].cpu().tolist()
        patch_size = getattr(layer, 'last_patch_size', 8)
        W_p = getattr(layer, 'last_W_p', 32)
        
        y_coords = np.array([ (idx // W_p) * patch_size + patch_size/2.0 for idx in indices])
        x_coords = np.array([ (idx % W_p) * patch_size + patch_size/2.0 for idx in indices])
        
        fig = plt.figure(figsize=(gt_pil.size[0]/100, gt_pil.size[1]/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(gt_pil)
        
        points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, len(x_coords))
        lc = LineCollection(segments, cmap='rainbow', norm=norm, alpha=0.7, linewidths=1.2)
        lc.set_array(np.arange(len(x_coords)))
        ax.add_collection(lc)
        
        # Start/End
        ax.scatter([x_coords[0]], [y_coords[0]], color='lime', s=30, zorder=5, edgecolors='black')
        ax.scatter([x_coords[-1]], [y_coords[-1]], color='red', s=30, zorder=5, edgecolors='black')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches=None, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    except Exception as e:
        print(f"Path draw failed: {e}")
        return gt_pil

# ---------------- CUSTOM MASK INDEXING ---------------- #
def index_custom_masks(mask_dir):
    print(f"Indexing masks in {mask_dir}...")
    categories = {'SMALL': [], 'MEDIUM': [], 'LARGE': []}
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for f in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, f)
        mask_img = Image.open(mask_path).convert('L')
        mask_np = np.array(mask_img)
        ratio = np.mean(mask_np) / 255.0
        
        if 0.01 < ratio <= 0.20:
            categories['SMALL'].append(mask_path)
        elif 0.20 < ratio <= 0.40:
            categories['MEDIUM'].append(mask_path)
        elif 0.40 < ratio <= 0.60:
            categories['LARGE'].append(mask_path)
            
    print(f"Index complete: SMALL({len(categories['SMALL'])}), MEDIUM({len(categories['MEDIUM'])}), LARGE({len(categories['LARGE'])})")
    return categories

# ---------------- MASK LOADING ---------------- #
def get_custom_mask(indexed_masks, cat, h, w):
    if not indexed_masks[cat]:
        # Fallback if a category is empty (should not happen with 12k masks)
        return torch.zeros((1, 1, h, w))
        
    mask_path = random.choice(indexed_masks[cat])
    mask_img = Image.open(mask_path).convert('L').resize((w, h), Image.NEAREST)
    mask_tensor = torchvision.transforms.functional.to_tensor(mask_img).float()
    return mask_tensor.unsqueeze(0)


# ---------------- POSTPROCESS ---------------- #
def postprocess(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


# ---------------- ORIGINAL METRICS ---------------- #
def calc_psnr_ssim(gt, pre):
    pre = pre.clamp(0, 1) * 255.0
    pre = pre.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]

    gt = gt.clamp(0, 1) * 255.0
    gt = gt.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]

    psnr = min(100, compare_psnr(gt, pre, data_range=255))
    ssim = compare_ssim(gt, pre, channel_axis=-1, data_range=255)

    return psnr, ssim


# ---------------- CORRECT MASKED METRICS ---------------- #
def calc_psnr_ssim_masked(gt, pre, mask):
    pre = pre.clamp(0, 1) * 255.0
    pre = pre.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]

    gt = gt.clamp(0, 1) * 255.0
    gt = gt.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]

    mask_np = mask[0, 0].cpu().numpy() > 0

    gt_masked = gt[mask_np]
    pre_masked = pre[mask_np]

    psnr = min(100, compare_psnr(gt_masked, pre_masked, data_range=255))

    # approximate SSIM on masked region
    gt_masked_img = gt * mask_np[..., None]
    pre_masked_img = pre * mask_np[..., None]

    ssim = compare_ssim(gt_masked_img, pre_masked_img,
                        channel_axis=-1, data_range=255)

    return psnr, ssim


# ---------------- LPIPS NORMALIZATION ---------------- #
def normalize_lpips(x):
    return (x * 2) - 1


# ---------------- MAIN ---------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./checkpoints')
    parser.add_argument('--output', type=str, default='./evaluation_results')
    args = parser.parse_args()

    config = Config(os.path.join(args.path, 'config.yml'))
    config.PATH = args.path
    config.MODE = 2
    config.MODEL = 2
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LPIPS
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    loss_fn_vgg.eval()

    # Model
    model = InpaintingModel(config).to(config.DEVICE)
    model.load()
    model.eval()

    test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
                           augment=False, training=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Index Custom Masks
    mask_dir = "/mnt/datadrive/inpaint/iregularmask/test_mask/mask/testing_mask_dataset"
    indexed_masks = index_custom_masks(mask_dir)

    categories = ['SMALL', 'MEDIUM', 'LARGE']

    stats = {
        cat: {
            'name': [],
            'psnr_full': [], 'ssim_full': [],
            'psnr_masked': [], 'ssim_masked': [],
            'l1_full': [], 'l1_masked': [],
            'lpips': []
        } for cat in categories
    }

    create_dir(args.output)

    fid_real_dirs = {cat: os.path.join(args.output, f'fid_real_{cat}') for cat in categories}
    fid_fake_dirs = {cat: os.path.join(args.output, f'fid_fake_{cat}') for cat in categories}
    visuals_dir = os.path.join(args.output, '5_image_grid')

    for cat in categories:
        create_dir(fid_real_dirs[cat])
        create_dir(fid_fake_dirs[cat])
        create_dir(os.path.join(visuals_dir, cat))

    # ---------------- LOOP ---------------- #
    for cat in categories:
        print(f"\nEvaluating {cat}")

        for index, items in enumerate(test_loader):
            images, _ = items
            images = images.to(config.DEVICE)

            h, w = images.shape[2], images.shape[3]
            masks = get_custom_mask(indexed_masks, cat, h, w).to(config.DEVICE)

            with torch.no_grad():
                outputs_img = model(images, masks)

            outputs_merged = (outputs_img * masks) + (images * (1 - masks))

            # -------- METRICS -------- #
            psnr_full, ssim_full = calc_psnr_ssim(images, outputs_merged)
            psnr_masked, ssim_masked = calc_psnr_ssim_masked(images, outputs_merged, masks)

            l1_full = F.l1_loss(outputs_merged, images).item()

            masked_diff = torch.abs(outputs_merged - images) * masks
            l1_masked = (masked_diff.sum() / (masks.sum() * images.shape[1] + 1e-8)).item()

            lpips_val = loss_fn_vgg(
                normalize_lpips(outputs_merged),
                normalize_lpips(images)
            ).item()

            file_name = test_dataset.load_name(index)

            stats[cat]['name'].append(file_name)
            stats[cat]['psnr_full'].append(psnr_full)
            stats[cat]['ssim_full'].append(ssim_full)
            stats[cat]['psnr_masked'].append(psnr_masked)
            stats[cat]['ssim_masked'].append(ssim_masked)
            stats[cat]['l1_full'].append(l1_full)
            stats[cat]['l1_masked'].append(l1_masked)
            stats[cat]['lpips'].append(lpips_val)

            # --- IMAGE SAVING ---
            gt_img_pil = Image.fromarray(postprocess(images)[0].cpu().numpy().astype(np.uint8))
            
            # 1. GT + Mask
            masked_input = (images * (1 - masks)) + masks
            gt_mask_pil = Image.fromarray(postprocess(masked_input)[0].cpu().numpy().astype(np.uint8))
            
            # 2. Mamba Path
            path_pil = get_mamba_path_image(model, gt_img_pil)
            
            # 3. Predicted (Raw)
            pred_raw_pil = Image.fromarray(postprocess(outputs_img)[0].cpu().numpy().astype(np.uint8))
            
            # 4. Merged (Composite)
            pred_merged_pil = Image.fromarray(postprocess(outputs_merged)[0].cpu().numpy().astype(np.uint8))
            
            # Concatenate
            grid = Image.new('RGB', (w * 5, h))
            grid.paste(gt_img_pil, (0, 0))
            grid.paste(gt_mask_pil, (w, 0))
            grid.paste(path_pil, (w * 2, 0))
            grid.paste(pred_raw_pil, (w * 3, 0))
            grid.paste(pred_merged_pil, (w * 4, 0))
            
            save_name = f"{cat}_{file_name.split('.')[0]}_{psnr_masked:.2f}.png"
            grid.save(os.path.join(visuals_dir, cat, save_name))

            # save for FID
            gt_img_pil.save(os.path.join(fid_real_dirs[cat], file_name))
            pred_merged_pil.save(os.path.join(fid_fake_dirs[cat], file_name))

    # ---------------- SAVE CSV ---------------- #
    for cat in categories:
        csv_path = os.path.join(args.output, f'metrics_{cat}.csv')

        fid_score = fid.compute_fid(fid_real_dirs[cat], fid_fake_dirs[cat])

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([
                'Image',
                'PSNR_full', 'SSIM_full',
                'PSNR_masked', 'SSIM_masked',
                'L1_full', 'L1_masked',
                'LPIPS'
            ])

            for i in range(len(stats[cat]['name'])):
                writer.writerow([
                    stats[cat]['name'][i],
                    stats[cat]['psnr_full'][i],
                    stats[cat]['ssim_full'][i],
                    stats[cat]['psnr_masked'][i],
                    stats[cat]['ssim_masked'][i],
                    stats[cat]['l1_full'][i],
                    stats[cat]['l1_masked'][i],
                    stats[cat]['lpips'][i]
                ])

            writer.writerow([])

            # averages
            avg_psnr_full = np.mean(stats[cat]['psnr_full'])
            avg_ssim_full = np.mean(stats[cat]['ssim_full'])
            avg_psnr_masked = np.mean(stats[cat]['psnr_masked'])
            avg_ssim_masked = np.mean(stats[cat]['ssim_masked'])
            avg_l1_full = np.mean(stats[cat]['l1_full'])
            avg_l1_masked = np.mean(stats[cat]['l1_masked'])
            avg_lpips = np.mean(stats[cat]['lpips'])

            writer.writerow([
                'AVERAGE_FULL',
                avg_psnr_full, avg_ssim_full,
                '-', '-',
                avg_l1_full, '-',
                avg_lpips
            ])

            writer.writerow([
                'AVERAGE_MASKED (CORRECT)',
                '-', '-',
                avg_psnr_masked, avg_ssim_masked,
                '-', avg_l1_masked,
                avg_lpips,
                f"FID: {fid_score:.4f}"
            ])

        print(f"{cat} done. FID: {fid_score:.4f}")

    print("Evaluation complete!")


if __name__ == '__main__':
    main()