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

def get_mask_for_category(h, w, cat):
    if cat == 'SMALL':
        target_range = (0.01, 0.20)
        max_parts, max_brush = 10, 20
    elif cat == 'MEDIUM':
        target_range = (0.20, 0.40)
        max_parts, max_brush = 15, 30
    elif cat == 'LARGE':
        target_range = (0.40, 0.60)
        max_parts, max_brush = 25, 45
    else:
        target_range = (0.01, 0.60)
        max_parts, max_brush = 15, 24
        
    while True:
        mask = generate_stroke_mask([h, w], max_parts=max_parts, maxBrushWidth=max_brush)
        mask = (mask > 0).astype(np.uint8) * 255
        ratio = np.mean(mask) / 255.0
        if target_range[0] < ratio <= target_range[1]:
            mask_img = Image.fromarray(mask.squeeze().astype(np.uint8), mode='L')
            mask_tensor = torchvision.transforms.functional.to_tensor(mask_img).float()
            return mask_tensor.unsqueeze(0)

def postprocess(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

def calc_psnr_ssim(gt, pre):
    pre = pre.clamp_(0, 1) * 255.0
    pre = pre.permute(0, 2, 3, 1)
    pre = pre.detach().cpu().numpy().astype(np.uint8)[0]
    gt = gt.clamp_(0, 1) * 255.0
    gt = gt.permute(0, 2, 3, 1)
    gt = gt.cpu().detach().numpy().astype(np.uint8)[0]
    psnr = min(100, compare_psnr(gt, pre, data_range=255))
    ssim = compare_ssim(gt, pre, multichannel=True, channel_axis=-1, data_range=255)
    return psnr, ssim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./checkpoints', help='model checkpoints path')
    parser.add_argument('--output', type=str, default='./evaluation_results', help='path to the output directory')
    args = parser.parse_args()

    config_path = os.path.join(args.path, 'config.yml')
    if not os.path.exists(config_path):
        import shutil
        print(f"Config not found in {args.path}, copying default ./config.yml...")
        shutil.copyfile('./config.yml', config_path)

    config = Config(config_path)
    config.PATH = args.path
    config.MODE = 2
    config.MODEL = 2
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize metrics
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    loss_fn_vgg.eval()
    transf = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    # Load Model
    model = InpaintingModel(config).to(config.DEVICE)
    model.load()
    model.eval()
    
    # Load Dataset
    test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Mask Categories Setup
    categories = ['SMALL', 'MEDIUM', 'LARGE', 'OTHER']
    stats = {cat: {'name': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []} for cat in categories}
    
    # Output Directories
    create_dir(args.output)
    fid_real_dirs = {cat: os.path.join(args.output, f'fid_real_{cat}') for cat in categories}
    fid_fake_dirs = {cat: os.path.join(args.output, f'fid_fake_{cat}') for cat in categories}
    results_dir = os.path.join(args.output, '5_image_format')
    
    for cat in categories:
        create_dir(fid_real_dirs[cat])
        create_dir(fid_fake_dirs[cat])
    create_dir(results_dir)
    
    print(f"Starting evaluation of {len(test_dataset)} images across 3 categories...")

    eval_categories = ['SMALL', 'MEDIUM', 'LARGE']
    for cat in eval_categories:
        print(f"\n--- Evaluating category: {cat} ---")
        for index, items in enumerate(test_loader):
            images, _ = items
            images = images.to(config.DEVICE)
            
            h, w = images.shape[2], images.shape[3]
            masks = get_mask_for_category(h, w, cat).to(config.DEVICE)
            
            with torch.no_grad():
                outputs_img = model(images, masks)
            
            outputs_merged = (outputs_img * masks) + (images * (1 - masks))
            
            # Metrics Calculation
            psnr, ssim = calc_psnr_ssim(images, outputs_merged)
            # Calculate L1 only on the masked region
            l1_error = F.l1_loss(outputs_merged, images, reduction='sum')
            l1 = (l1_error / (masks.sum() * images.shape[1] + 1e-8)).item()
            
            pl = loss_fn_vgg(transf(outputs_merged[0].cpu()).to(config.DEVICE), transf(images[0].cpu()).to(config.DEVICE)).item()
            
            file_name_for_stats = test_dataset.load_name(index)
            stats[cat]['name'].append(file_name_for_stats)
            stats[cat]['psnr'].append(psnr)
            stats[cat]['ssim'].append(ssim)
            stats[cat]['l1'].append(l1)
            stats[cat]['lpips'].append(pl)
            
            # Generate the 5 Image Format
            gt_img_pil = Image.fromarray(postprocess(images)[0].cpu().numpy().astype(np.uint8))
            mask_pil = Image.fromarray((masks[0][0].cpu().numpy() * 255).astype(np.uint8)).convert('RGB')
            masked_inputs = (images * (1 - masks)) + masks
            gt_masked_pil = Image.fromarray(postprocess(masked_inputs)[0].cpu().numpy().astype(np.uint8))
            pred_img_pil = Image.fromarray(postprocess(outputs_img)[0].cpu().numpy().astype(np.uint8))
            pred_merged_pil = Image.fromarray(postprocess(outputs_merged)[0].cpu().numpy().astype(np.uint8))
            
            widths, heights = zip(*(i.size for i in [gt_img_pil, mask_pil, gt_masked_pil, pred_img_pil, pred_merged_pil]))
            total_width = sum(widths)
            max_height = max(heights)
            
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in [gt_img_pil, mask_pil, gt_masked_pil, pred_img_pil, pred_merged_pil]:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]
                
            # Naming format: cat_imageID_PSNR.png
            file_name = test_dataset.load_name(index)
            base_name = os.path.splitext(file_name)[0]
            save_name = f"{cat}_{base_name}_{psnr:.2f}.png"
            new_im.save(os.path.join(results_dir, save_name))
            
            # Save independent images for FID Calculation later
            gt_img_pil.save(os.path.join(fid_real_dirs[cat], file_name))
            pred_merged_pil.save(os.path.join(fid_fake_dirs[cat], file_name))

            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1} / {len(test_dataset)} images for {cat}")

    print("Computing FID scores and saving CSV files...")
    
    for cat in ['SMALL', 'MEDIUM', 'LARGE']:
        count = len(stats[cat]['psnr'])
        if count > 0:
            fid_score = fid.compute_fid(fid_real_dirs[cat], fid_fake_dirs[cat])
            
            avg_psnr = np.mean(stats[cat]['psnr'])
            avg_ssim = np.mean(stats[cat]['ssim'])
            avg_l1 = np.mean(stats[cat]['l1'])
            avg_lpips = np.mean(stats[cat]['lpips'])
            
            csv_path = os.path.join(args.output, f'metrics_{cat.lower()}.csv')
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image', 'PSNR', 'SSIM', 'L1', 'LPIPS'])
                
                for i in range(count):
                    writer.writerow([
                        stats[cat]['name'][i],
                        stats[cat]['psnr'][i],
                        stats[cat]['ssim'][i],
                        stats[cat]['l1'][i],
                        stats[cat]['lpips'][i]
                    ])
                
                writer.writerow([])
                writer.writerow(['AVERAGE/TOTAL', avg_psnr, avg_ssim, avg_l1, avg_lpips, f"FID: {fid_score:.4f}"])
                
            print(f"{cat} ({count} imgs) saved to {csv_path} | Avg PSNR: {avg_psnr:.4f} | FID: {fid_score:.4f}")
        else:
            print(f"{cat} (0 imgs) | Skipped")

    print(f"Evaluation complete! Results saved at: {args.output}")

if __name__ == '__main__':
    main()
