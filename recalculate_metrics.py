import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import csv
import warnings
warnings.filterwarnings('ignore')

def calc_psnr_ssim(gt_np, pre_np):
    psnr = min(100, compare_psnr(gt_np, pre_np, data_range=255))
    ssim = compare_ssim(gt_np, pre_np, multichannel=True, channel_axis=-1, data_range=255)
    return psnr, ssim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_fn_vgg.eval()
    
    transf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    to_tensor = torchvision.transforms.ToTensor()
    
    input_dir = './evaluation_results/5_image_format'
    output_dir = './evaluation_results'
    
    categories = ['SMALL', 'MEDIUM', 'LARGE']
    stats = {cat: {'name': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []} for cat in categories}
    
    file_list = glob.glob(os.path.join(input_dir, '*.png'))
    total_files = len(file_list)
    print(f"Found {total_files} files in {input_dir}")
    if total_files == 0:
        return
        
    for idx, fp in enumerate(file_list):
        filename = os.path.basename(fp)
        
        # Name format: cat_imageID_PSNR.png
        parts = filename.split('_')
        cat = parts[0]
        if cat not in categories:
            continue
            
        img = Image.open(fp).convert('RGB')
        W, H = img.size
        w = W // 5
        
        gt_img = img.crop((0, 0, w, H))
        mask_img = img.crop((w, 0, 2*w, H)).convert('L')
        pred_merged_img = img.crop((4*w, 0, W, H))
        
        gt_np = np.array(gt_img)
        pre_np = np.array(pred_merged_img)
        
        psnr, ssim = calc_psnr_ssim(gt_np, pre_np)
        
        gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device)
        pre_tensor = to_tensor(pred_merged_img).unsqueeze(0).to(device)
        
        mask_tensor = to_tensor(mask_img).unsqueeze(0).to(device) 
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Recalculate L1 on masked region ONLY
        l1_error = F.l1_loss(pre_tensor, gt_tensor, reduction='sum')
        l1 = (l1_error / (mask_tensor.sum() * 3 + 1e-8)).item()
        
        gt_lpips = transf(gt_img).unsqueeze(0).to(device)
        pre_lpips = transf(pred_merged_img).unsqueeze(0).to(device)
        pl = loss_fn_vgg(pre_lpips, gt_lpips).item()
        
        original_base_name = "_".join(parts[1:-1]) 
        
        stats[cat]['name'].append(original_base_name)
        stats[cat]['psnr'].append(psnr)
        stats[cat]['ssim'].append(ssim)
        stats[cat]['l1'].append(l1)
        stats[cat]['lpips'].append(pl)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_files} files")
            
    print("Saving updated CSVs...")
    for cat in categories:
        count = len(stats[cat]['psnr'])
        if count > 0:
            avg_psnr = np.mean(stats[cat]['psnr'])
            avg_ssim = np.mean(stats[cat]['ssim'])
            avg_l1 = np.mean(stats[cat]['l1'])
            avg_lpips = np.mean(stats[cat]['lpips'])
            
            # Fetch previous FID score if possible
            fid_str = "FID: N/A"
            old_csv = os.path.join(output_dir, f'metrics_{cat.lower()}.csv')
            if os.path.exists(old_csv):
                with open(old_csv, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        last_line = lines[-1].strip().split(',')
                        if len(last_line) >= 6 and 'FID:' in last_line[5]:
                            fid_str = last_line[5]
            
            csv_path = os.path.join(output_dir, f'updated_metrics_{cat.lower()}.csv')
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image', 'PSNR', 'SSIM', 'L1', 'LPIPS'])
                
                # Sort by filename to keep it consistent
                combined = list(zip(stats[cat]['name'], stats[cat]['psnr'], stats[cat]['ssim'], stats[cat]['l1'], stats[cat]['lpips']))
                combined.sort(key=lambda x: x[0])
                
                for row in combined:
                    writer.writerow(row)
                
                writer.writerow([])
                writer.writerow(['AVERAGE/TOTAL', avg_psnr, avg_ssim, avg_l1, avg_lpips, fid_str])
                
            print(f"{cat} ({count} imgs) saved to {csv_path} | Avg L1: {avg_l1:.4f} | {fid_str}")

if __name__ == '__main__':
    main()
