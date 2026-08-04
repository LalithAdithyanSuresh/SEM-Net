import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import Dataset
from src.models import InpaintingModel
from src.utils import create_dir
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import torchvision
from PIL import Image
from cleanfid import fid
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import io

def postprocess(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

def calc_psnr_ssim(gt, pre):
    # gt and pre are 3-channel RGB tensors [1, 3, H, W]
    pre = pre.clamp(0, 1) * 255.0
    pre = pre.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    gt = gt.clamp(0, 1) * 255.0
    gt = gt.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    psnr = min(100, compare_psnr(gt, pre, data_range=255))
    ssim = compare_ssim(gt, pre, channel_axis=-1, data_range=255)
    return psnr, ssim


def normalize_lpips(x):
    return (x * 2) - 1

def index_custom_masks(mask_dir):
    print(f"Indexing masks in {mask_dir}...")
    categories = {'SMALL': [], 'MEDIUM': [], 'LARGE': []}
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files.sort()
    
    for f in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, f)
        try:
            mask_img = Image.open(mask_path).convert('L')
            mask_np = np.array(mask_img)
            ratio = np.mean(mask_np) / 255.0
            
            if 0.01 < ratio <= 0.20:
                categories['SMALL'].append((f, mask_path))
            elif 0.20 < ratio <= 0.40:
                categories['MEDIUM'].append((f, mask_path))
            elif 0.40 < ratio <= 0.60:
                categories['LARGE'].append((f, mask_path))
        except Exception as e:
            print(f"Skip {f}: {e}")
            
    print(f"Index complete: SMALL({len(categories['SMALL'])}), MEDIUM({len(categories['MEDIUM'])}), LARGE({len(categories['LARGE'])})")
    return categories

def get_custom_mask(indexed_masks, cat, index, h, w):
    mask_list = indexed_masks[cat]
    if not mask_list:
        return torch.zeros((1, 1, h, w)), "no_mask"
    
    # Use rotation over all available masks in the category
    mask_name, mask_path = mask_list[index % len(mask_list)]
    mask_img = Image.open(mask_path).convert('L').resize((w, h), Image.NEAREST)
    mask_tensor = torchvision.transforms.functional.to_tensor(mask_img).float()
    mask_id = os.path.splitext(mask_name)[0]
    return mask_tensor.unsqueeze(0), mask_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./segmentCamino')
    parser.add_argument('--output', type=str, default='./evaluation_results_ffhq')
    parser.add_argument('--dataset_root', type=str, default='./dataset')
    args = parser.parse_args()

    config_path = os.path.join(args.path, 'config.yml')
    
    # Run prerequisites setup from setup_server.py (Steps 1-11 only, omitting Step 12 Segment Map generation)
    if os.path.exists("setup_server.py"):
        try:
            print("Checking prerequisites from setup_server.py...")
            import setup_server
            
            # Replicate setup_server's steps 1-11 setup without launching anything or running step 12
            setup_server.step_verify_workspace()
            setup_server.step_setup_cuda()
            setup_server.step_create_venv()
            
            # Setup pip setup environment values
            venv_pip = os.path.abspath(os.path.join("venv", "bin", "pip"))
            python_bin = os.path.abspath(os.path.join("venv", "bin", "python"))
            
            # Check and run step 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11
            import subprocess
            subprocess.check_call([venv_pip, "install", "--upgrade", "pip", "wheel"], stdout=subprocess.DEVNULL)
            if not setup_server.check_setuptools_numpy_installed():
                subprocess.check_call([venv_pip, "install", "setuptools<82", "numpy<2"])
            if not setup_server.check_pytorch_installed():
                subprocess.check_call([venv_pip, "install", "torch==2.1.2", "torchvision==0.16.2", "--extra-index-url", "https://download.pytorch.org/whl/cu121"])
            if not setup_server.check_ninja_packaging_installed():
                subprocess.check_call([venv_pip, "install", "packaging", "ninja"])
            if not setup_server.check_mamba_installed():
                subprocess.check_call([venv_pip, "install", "causal-conv1d==1.1.3.post1", "mamba-ssm==1.1.3.post1", "--no-build-isolation", "-v"])
            
            subprocess.check_call([venv_pip, "install", "-r", "requirements.txt"], stdout=subprocess.DEVNULL)
            setup_server.step_download_ops()
            
            if not setup_server.check_dcnv3_compiled():
                ops_dir = os.path.abspath(os.path.join("src", "ops_dcnv3"))
                setup_server.patch_pytorch_boxing_header()
                subprocess.check_call(["sh", "make.sh"], cwd=ops_dir)
                
            print("Prerequisites verified successfully.")
        except Exception as e:
            print(f"Prerequisite verification finished with warning: {e}. Attempting execution anyway...")
            
    config = Config(config_path)
    config.PATH = args.path
    config.MODE = 2
    config.MODEL = 2
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override image flists for evaluate script
    config.TEST_INPAINT_IMAGE_FLIST = os.path.join(args.dataset_root, 'test')
    config.TEST_MASK_FLIST = os.path.join(args.dataset_root, 'masks')

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
    mask_dir = os.path.join(args.dataset_root, 'masks')
    indexed_masks = index_custom_masks(mask_dir)

    categories = ['SMALL', 'MEDIUM', 'LARGE']
    create_dir(args.output)

    # Process each category
    for cat in categories:
        print(f"\nEvaluating category: {cat}")
        cat_output_dir = os.path.join(args.output, cat)
        create_dir(cat_output_dir)
        
        csv_path = os.path.join(args.output, f'metrics_{cat}.csv')
        stats = {'name': [], 'mask_id': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []}

        for index, items in enumerate(tqdm(test_loader, desc=f"Eval {cat}")):
            # Retrieve items
            # In src/dataset.py: load_item returns: to_tensor(img), to_tensor(mask), to_tensor(seg_map)
            images, _, seg_maps = items
            images = images.to(config.DEVICE)
            seg_maps = seg_maps.to(config.DEVICE)
            h, w = images.shape[2], images.shape[3]
            
            # Load custom mask in rotation
            masks, mask_id = get_custom_mask(indexed_masks, cat, index, h, w)
            masks = masks.to(config.DEVICE)

            with torch.no_grad():
                outputs_img = model(images, masks, seg_maps=seg_maps)

            outputs_merged = (outputs_img * masks) + (images * (1 - masks))

            # Metrics
            psnr, ssim = calc_psnr_ssim(images, outputs_merged)
            l1_val = F.l1_loss(outputs_merged, images, reduction='mean').item()
            lpips_val = loss_fn_vgg(normalize_lpips(outputs_merged), normalize_lpips(images)).item()

            file_name = test_dataset.load_name(index)
            image_id = os.path.splitext(file_name)[0]
            
            stats['name'].append(file_name)
            stats['mask_id'].append(mask_id)
            stats['psnr'].append(psnr)
            stats['ssim'].append(ssim)
            stats['l1'].append(l1_val)
            stats['lpips'].append(lpips_val)

            # Save generated image with exact naming convention: imageID_maskID_PSNR.png
            pred_merged_pil = Image.fromarray(postprocess(outputs_merged)[0].cpu().numpy().astype(np.uint8))
            save_name = f"{image_id}_{mask_id}_{psnr:.2f}.png"
            pred_merged_pil.save(os.path.join(cat_output_dir, save_name))

        # Write CSV for this category
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'MaskID', 'PSNR', 'SSIM', 'L1', 'LPIPS'])
            for i in range(len(stats['name'])):
                writer.writerow([stats['name'][i], stats['mask_id'][i], f"{stats['psnr'][i]:.4f}", f"{stats['ssim'][i]:.4f}", f"{stats['l1'][i]:.6f}", f"{stats['lpips'][i]:.6f}"])
            
            writer.writerow([])
            writer.writerow(['AVERAGE', '', f"{np.mean(stats['psnr']):.4f}", f"{np.mean(stats['ssim']):.4f}", f"{np.mean(stats['l1']):.6f}", f"{np.mean(stats['lpips']):.6f}"])

        print(f"Category {cat} finalized. Average PSNR: {np.mean(stats['psnr']):.2f}")

    print(f"All done! Results saved in {args.output}")

if __name__ == '__main__':
    main()
