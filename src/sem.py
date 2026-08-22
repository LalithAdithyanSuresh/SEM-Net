import os

if 'TORCH_HOME' not in os.environ:
    os.environ['TORCH_HOME'] = os.path.abspath('./tmp/torch_cache')
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

import json
import numpy as np
import torch
import torch.hub
torch.hub.set_dir(os.path.join(os.environ['TORCH_HOME'], 'hub'))

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
try:
    import wandb
except ImportError:
    wandb = None
from cv2 import circle
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import torchvision
import time

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''
import requests
import sys

import threading

# Assume the C2 URL is passed via environment variable (or default to port 443 of VPS)
C2_SERVER_URL = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')
C2_SESSION    = os.environ.get('C2_SESSION', 'segment')

# Automatically route file uploads to the optimized files subdomain if C2 is on the main domain
default_files_url = C2_SERVER_URL
if 'lalithadithyan.dev' in C2_SERVER_URL and 'files.' not in C2_SERVER_URL:
    default_files_url = C2_SERVER_URL.replace('lalithadithyan.dev', 'files.lalithadithyan.dev')

FILES_SERVER_URL = os.environ.get('FILES_SERVER_URL', default_files_url)

def upload_file_chunked(file_path, server_url, session_id, chunk_size=10 * 1024 * 1024, target_filename=None):
    # Network uploads disabled
    return False


class sem():
    def __init__(self, config):
        self.config = config


        if config.MODEL == 2:
            model_name = 'inpaint'

        self.debug = False
        self.model_name = model_name

        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # Pre-download pretrained weights sequentially on Rank 0 to prevent download conflicts
        if dist.is_initialized() and config.WORLD_SIZE > 1:
            if config.RANK == 0:
                print("Pre-downloading pretrained model weights on Rank 0 to avoid write conflicts...")
                import torchvision.models as models
                # Trigger downloads for VGG19 (used by PerceptualLoss) and VGG16 (used by LPIPS)
                _ = models.vgg19(pretrained=True)
                _ = lpips.LPIPS(net='vgg')
            dist.barrier()  # All other ranks wait for Rank 0 to finish downloading

        # Initialize models collectively across all ranks (DDP requires simultaneous instantiation)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        # datasets
        if self.config.MODEL == 2:
            self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)


        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')
        # Persist epoch across restarts so it doesn't reset to 0 on resume
        self.epoch_state_file = os.path.join(config.PATH, 'epoch_state.json')

        # Best model tracking
        self.best_epoch_psnr = -1.0
        self.top5_best_models = []

    def load(self):


        if self.config.MODEL == 2:
            self.inpaint_model.load()


    def save(self):
 
        if self.config.MODEL == 2:
            self.inpaint_model.save()


    def train(self):
        
        if self.config.WORLD_SIZE > 1:
            sampler = DistributedSampler(self.train_dataset, num_replicas=self.config.WORLD_SIZE, rank=self.config.RANK)
        else:
            sampler = None

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=max(1, self.config.BATCH_SIZE // self.config.WORLD_SIZE), # Split batch across processes
            num_workers=6,            # Optimized for 2-GPU DDP
            drop_last=True,
            shuffle=(sampler is None),
            pin_memory=True,
            sampler=sampler
        )


        # --- Epoch Persistence: load saved epoch so restarts don't reset to 0 ---
        epoch = 0
        if os.path.exists(self.epoch_state_file):
            try:
                with open(self.epoch_state_file, 'r') as _ef:
                    _saved = json.load(_ef)
                    epoch = int(_saved.get('epoch', 0))
                    print(f'[RESUME] Continuing from epoch {epoch}, iteration {self.inpaint_model.iteration}')
            except Exception:
                pass
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        
        # --- Local accumulator: collect every iteration's metrics, flush every 300 iters ---
        _METRIC_KEYS = ['gen_loss', 'dis_loss', 'l1_loss', 'perceptual_loss',
                        'style_loss', 'sym_loss', 'gan_loss', 'psnr', 'mae']
        _metric_buf = {k: [] for k in _METRIC_KEYS}
        _metric_buf_epoch = []
        
        try:
            while(keep_training):
                epoch += 1
                self.best_epoch_psnr = -1.0
                if self.config.RANK == 0:
                    print(f"Training epoch: {epoch}")
                    # Show progress within the current epoch
                    progbar = Progbar(len(train_loader), width=20, stateful_metrics=['epoch', 'iter'])
                
                if sampler is not None:
                    sampler.set_epoch(epoch)
                for items in train_loader:
                    iteration = self.inpaint_model.iteration
                    self.inpaint_model.train()


                    if model == 2:
                        if len(items) >= 3:
                            images, masks, seg_maps = self.cuda(*items)
                        else:
                            images, masks = self.cuda(*items[:2])
                            seg_maps = None

                        outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, gen_symmetry_loss = self.inpaint_model.process(images, masks, seg_maps)
                        outputs_merged = (outputs_img * masks) + (images * (1-masks))

                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                        mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                        logs.append(('psnr', psnr.item()))
                        logs.append(('mae', mae.item()))

                        self.inpaint_model.backward(gen_loss, dis_loss)

                        # print nvidia-smi output after the first iteration is processed
                        if self.config.RANK == 0 and self.inpaint_model.iteration == 1:
                            print("\n=== nvidia-smi (GPU allocation after 1st iteration) ===")
                            import subprocess
                            try:
                                res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                                print(res.stdout)
                            except Exception as e:
                                print(f"Could not run nvidia-smi: {e}")
                            print("=========================================================\n")

                        # --- Accumulate every iteration into local buffer ---
                        _metric_buf['gen_loss'].append(float(gen_loss))
                        _metric_buf['dis_loss'].append(float(dis_loss))
                        _metric_buf['l1_loss'].append(float(gen_l1_loss))
                        _metric_buf['perceptual_loss'].append(float(gen_content_loss))
                        _metric_buf['style_loss'].append(float(gen_style_loss))
                        _metric_buf['sym_loss'].append(float(gen_symmetry_loss))
                        _metric_buf['gan_loss'].append(float(gen_gan_loss))
                        _metric_buf['psnr'].append(float(psnr.item()))
                        _metric_buf['mae'].append(float(mae.item()))
                        _metric_buf_epoch.append(epoch)

                        # Reset buffers for the next 300-iter window
                        if iteration > 0 and iteration % 300 == 0:
                            _metric_buf = {k: [] for k in _METRIC_KEYS}
                            _metric_buf_epoch = []
                        iteration = self.inpaint_model.iteration



                    if iteration >= max_iteration:
                        keep_training = False
                        break

                    logs = [
                        ("iter", iteration),
                        ("gLoss", gen_loss.item()),
                        ("dLoss", dis_loss.item()),
                    ] + [ (k, v.item() if hasattr(v, 'item') else v) for k, v in logs ]

                    if self.config.RANK == 0:
                        progbar.add(1, values=logs if self.config.WORLD_SIZE == 1 else [x for x in logs if x[0] != 'iter'])
                    if iteration % 10 == 0 and wandb is not None and wandb.run is not None:
                            wandb.log({'gen_loss': gen_loss, 'l1_loss': gen_l1_loss, 'style_loss': gen_style_loss,
                                       'perceptual loss': gen_content_loss, 'gen_gan_loss': gen_gan_loss,
                                       'gen_symmetry_loss': gen_symmetry_loss,
                                       'dis_loss': dis_loss}, step=iteration)
             


                    val_milestones = {1, 10, 50, 100, 200, 400, 500, 1000, 1500}
                    if self.config.RANK == 0 and ((iteration in val_milestones) or (iteration > 0 and iteration % 2000 == 0)):
                        create_dir(self.results_path)
                        path_val = os.path.join(self.results_path, self.model_name, 'validation')
                        create_dir(path_val)
                        
                        self.inpaint_model.eval()
                        import matplotlib.pyplot as plt
                        import io
                        import numpy as np
                        from torch.utils.data import Subset

                        # Use test_dataset if populated, otherwise fallback to train_dataset so validation never fails
                        eval_ds     = self.test_dataset if len(self.test_dataset) > 0 else self.train_dataset
                        n_test      = len(eval_ds)
                        first_idx   = list(range(min(5, n_test)))
                        last_idx    = list(range(max(0, n_test - 5), n_test))
                        all_indices = list(dict.fromkeys(first_idx + last_idx))
                        val_loader  = DataLoader(dataset=Subset(eval_ds, all_indices),
                                                 batch_size=1, num_workers=0, shuffle=False)

                        # ── Helper: render full scan-path panel (lines, for known+hole) ─
                        def _draw_path_panel(scan_orders, mask_np, bg_pil, patch_size, img_size):
                            from matplotlib.collections import LineCollection
                            fig = plt.figure(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
                            ax  = fig.add_axes([0, 0, 1, 1])
                            ax.axis('off')
                            ax.imshow(np.array(bg_pil))
                            if scan_orders:
                                y_all = np.array([r * patch_size + patch_size/2.0 for r, c in scan_orders])
                                x_all = np.array([c * patch_size + patch_size/2.0 for r, c in scan_orders])
                                pts  = np.array([x_all, y_all]).T.reshape(-1, 1, 2)
                                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                                lc   = LineCollection(segs, cmap='rainbow',
                                                      norm=plt.Normalize(0, len(x_all)),
                                                      alpha=0.80, linewidths=1.5)
                                lc.set_array(np.arange(len(x_all)))
                                ax.add_collection(lc)
                                if len(x_all) > 0:
                                    ax.scatter([x_all[0]],  [y_all[0]],  color='lime', s=45, zorder=5, edgecolors='black')
                                    ax.scatter([x_all[-1]], [y_all[-1]], color='red',  s=45, zorder=5, edgecolors='black')
                                    step = max(1, len(x_all) // 15)
                                    for k in range(0, len(x_all)-1, step):
                                        ddx = x_all[k+1]-x_all[k]; ddy = y_all[k+1]-y_all[k]
                                        dist = np.hypot(ddx, ddy)
                                        if dist > 0:
                                            ax.arrow(x_all[k], y_all[k],
                                                     (ddx/dist)*(patch_size*0.45), (ddy/dist)*(patch_size*0.45),
                                                     color='white', head_width=patch_size*0.4, alpha=1.0, zorder=6)
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', dpi=100)
                            plt.close(fig)
                            buf.seek(0)
                            return Image.open(buf).convert('RGB').resize(img_size)

                        # ── Helper: hole-only patch heatmap (no lines, plasma gradient) ─
                        def _draw_hole_heatmap(scan_orders, mask_np, img_size, patch_size):
                            H_px, W_px = img_size[1], img_size[0]
                            canvas = np.zeros((H_px, W_px, 3), dtype=np.uint8)

                            if not scan_orders:
                                return Image.fromarray(canvas)

                            H_m, W_m = mask_np.shape
                            cmap = plt.cm.plasma

                            hole_patches = []
                            for (p_i, p_j) in scan_orders:
                                cy = min(int(p_i * patch_size + patch_size // 2), H_m - 1)
                                cx = min(int(p_j * patch_size + patch_size // 2), W_m - 1)
                                if mask_np[cy, cx] > 0.5:
                                    hole_patches.append((p_i, p_j))

                            if not hole_patches:
                                return Image.new('RGB', img_size, (30, 30, 30))

                            n = len(hole_patches)
                            for local_rank, (p_i, p_j) in enumerate(hole_patches):
                                t   = 1.0 - (local_rank / max(n - 1, 1))
                                r, g, b, _ = cmap(t)
                                color = (int(r * 255), int(g * 255), int(b * 255))

                                y0 = int(p_i * patch_size)
                                x0 = int(p_j * patch_size)
                                y1 = min(y0 + patch_size, H_px)
                                x1 = min(x0 + patch_size, W_px)

                                canvas[y0:y1, x0:x1] = color

                                if patch_size > 2:
                                    dark = (max(color[0]-60, 0), max(color[1]-60, 0), max(color[2]-60, 0))
                                    canvas[y0, x0:x1]   = dark
                                    canvas[y1-1, x0:x1] = dark
                                    canvas[y0:y1, x0]   = dark
                                    canvas[y0:y1, x1-1] = dark

                            return Image.fromarray(canvas)

                        # ── Helper: hole lines overlaid on plasma heatmap ────────────
                        def _draw_hole_path_overlay(scan_orders, mask_np, img_size, patch_size):
                            from matplotlib.collections import LineCollection
                            H_px, W_px = img_size[1], img_size[0]
                            canvas = np.zeros((H_px, W_px, 3), dtype=np.uint8)

                            if not scan_orders:
                                return Image.fromarray(canvas)

                            H_m, W_m = mask_np.shape
                            cmap_plasma = plt.cm.plasma

                            hole_patches = []
                            for (p_i, p_j) in scan_orders:
                                cy = min(int(p_i * patch_size + patch_size // 2), H_m - 1)
                                cx = min(int(p_j * patch_size + patch_size // 2), W_m - 1)
                                if mask_np[cy, cx] > 0.5:
                                    hole_patches.append((p_i, p_j))

                            if not hole_patches:
                                return Image.new('RGB', img_size, (30, 30, 30))

                            n = len(hole_patches)
                            for local_rank, (p_i, p_j) in enumerate(hole_patches):
                                t = 1.0 - (local_rank / max(n - 1, 1))
                                r, g, b, _ = cmap_plasma(t)
                                color = (int(r*255), int(g*255), int(b*255))
                                y0, x0 = int(p_i*patch_size), int(p_j*patch_size)
                                y1, x1 = min(y0+patch_size, H_px), min(x0+patch_size, W_px)
                                canvas[y0:y1, x0:x1] = color

                            base_img = Image.fromarray(canvas)
                            fig = plt.figure(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
                            ax  = fig.add_axes([0, 0, 1, 1])
                            ax.axis('off')
                            ax.imshow(np.array(base_img))

                            y_h = np.array([r * patch_size + patch_size/2.0 for r, c in hole_patches])
                            x_h = np.array([c * patch_size + patch_size/2.0 for r, c in hole_patches])
                            pts  = np.array([x_h, y_h]).T.reshape(-1, 1, 2)
                            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                            lc   = LineCollection(segs, cmap='cool',
                                                  norm=plt.Normalize(0, n),
                                                  alpha=0.85, linewidths=1.2)
                            lc.set_array(np.arange(n))
                            ax.add_collection(lc)
                            if n > 0:
                                ax.scatter([x_h[0]],  [y_h[0]],  color='lime', s=40, zorder=5, edgecolors='black')
                                ax.scatter([x_h[-1]], [y_h[-1]], color='red',  s=40, zorder=5, edgecolors='black')

                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', dpi=100)
                            plt.close(fig)
                            buf.seek(0)
                            return Image.open(buf).convert('RGB').resize(img_size)

                        val_count = 0
                        val_psnr_list = []
                        for val_items in val_loader:
                            if len(val_items) >= 3:
                                val_images, val_masks, val_seg_maps = self.cuda(*val_items)
                            else:
                                val_images, val_masks = self.cuda(*val_items[:2])
                                val_seg_maps = None
                            val_inputs = (val_images * (1 - val_masks)) + val_masks
                            with torch.no_grad():
                                val_outputs_img = self.inpaint_model(val_images, val_masks, val_seg_maps)
                            
                            val_outputs_merged = (val_outputs_img * val_masks) + (val_images * (1 - val_masks))
                            val_psnr_val = self.psnr(self.postprocess(val_images), self.postprocess(val_outputs_merged)).item()
                            val_psnr_list.append(val_psnr_val)

                            # ── Extract scan path & attn layer (every image) ─────────
                            patch_size  = 1
                            scan_orders = None
                            attn_layer  = None
                            try:
                                if hasattr(self.inpaint_model.generator, 'module'):
                                    attn_layer = self.inpaint_model.generator.module.encoder_level1[0].attn
                                else:
                                    attn_layer = self.inpaint_model.generator.encoder_level1[0].attn
                                if getattr(attn_layer, 'last_scan_orders', None) is not None:
                                    scan_orders_tensor = attn_layer.last_scan_orders[0]
                                    patch_size  = getattr(attn_layer, 'last_patch_size', 1)
                                    W_p         = getattr(attn_layer, 'last_W_p', 256 // max(patch_size, 1))
                                    scan_orders = [(int(idx) // W_p, int(idx) % W_p)
                                                   for idx in scan_orders_tensor.cpu().tolist()]
                            except Exception as e:
                                pass

                            # ── PIL conversions ───────────────────────────────────────
                            gt_img_pil    = Image.fromarray(self.postprocess(val_images)[0].cpu().numpy().astype(np.uint8))
                            gt_mask_pil   = Image.fromarray(self.postprocess(val_inputs)[0].cpu().numpy().astype(np.uint8))
                            pred_img_pil  = Image.fromarray(self.postprocess(val_outputs_img)[0].cpu().numpy().astype(np.uint8))
                            pred_mask_pil = Image.fromarray(self.postprocess(val_outputs_merged)[0].cpu().numpy().astype(np.uint8))
                            img_size = gt_img_pil.size
                            mask_np  = val_masks[0, 0].cpu().float().numpy()

                            full_path_pil  = _draw_path_panel(scan_orders, mask_np, gt_mask_pil, patch_size, img_size)
                            hole_path_pil  = _draw_hole_heatmap(scan_orders, mask_np, img_size, patch_size)
                            hole_lines_pil = _draw_hole_path_overlay(scan_orders, mask_np, img_size, patch_size)

                            try:
                                import cv2
                                da_offset_pil = None
                                da_offset_map = getattr(attn_layer.da_scan, 'last_offset_map', None) \
                                                if attn_layer is not None else None
                                if da_offset_map is not None:
                                    off     = da_offset_map[0].cpu().float().numpy()
                                    G       = off.shape[-1] // 2
                                    mag     = np.sqrt(off[..., :G]**2 + off[..., G:]**2).mean(axis=-1)
                                    mag     = (mag - mag.min()) / (mag.max() + 1e-8)
                                    mag_u8  = (mag * 255).astype(np.uint8)
                                    mag_bgr = cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)
                                    da_offset_pil = Image.fromarray(
                                        cv2.cvtColor(mag_bgr, cv2.COLOR_BGR2RGB)).resize(img_size)
                                if da_offset_pil is None:
                                    da_offset_pil = Image.new('RGB', img_size, (80, 80, 80))
                            except Exception:
                                da_offset_pil = Image.new('RGB', img_size, (80, 80, 80))

                            # ── Segment Map ───────────────────────────────────────────
                            orig_idx  = all_indices[val_count]
                            img_path = eval_ds.data[orig_idx]
                            img_dir  = os.path.dirname(img_path)
                            img_name = eval_ds.load_name(orig_idx)
                            base_name, _ = os.path.splitext(img_name)
                            
                            seg_map_pil = None

                            # 1. Try rendering directly from loaded DataLoader val_seg_maps tensor
                            if val_seg_maps is not None:
                                try:
                                    seg_arr = val_seg_maps[0].cpu().numpy().squeeze().astype(np.uint8)
                                    if seg_arr.ndim == 2 and seg_arr.max() > 0:
                                        import cv2
                                        scaled_seg = (seg_arr.astype(np.float32) / float(seg_arr.max()) * 255.0).astype(np.uint8)
                                        color_seg  = cv2.applyColorMap(scaled_seg, cv2.COLORMAP_TURBO)
                                        seg_map_pil = Image.fromarray(cv2.cvtColor(color_seg, cv2.COLOR_BGR2RGB)).resize(img_size)
                                except Exception:
                                    pass

                            # 2. If tensor visualization wasn't available, search disk with expanded path candidates
                            if seg_map_pil is None:
                                seg_mask_path = None
                                possible_seg_paths = [
                                    os.path.join(img_dir + "_seg", f"{base_name}.png"),
                                    os.path.join(os.path.dirname(img_dir), os.path.basename(img_dir) + "_seg", f"{base_name}.png"),
                                    os.path.join("dataset", "train_seg", f"{base_name}.png"),
                                    os.path.join("dataset", "test_seg", f"{base_name}.png"),
                                    os.path.join(os.path.dirname(img_dir) + "_seg", os.path.basename(img_dir), f"{base_name}.png"),
                                    os.path.join(os.path.dirname(img_dir) + "_seg", f"{base_name}.png"),
                                    os.path.join(os.path.dirname(os.path.dirname(img_dir)) + "_seg", f"{base_name}.png"),
                                    os.path.join(os.path.dirname(os.path.dirname(img_dir)) + "_seg", os.path.basename(img_dir), f"{base_name}.png"),
                                ]
                                for pth in possible_seg_paths:
                                    if os.path.exists(pth):
                                        seg_mask_path = pth
                                        break

                                if seg_mask_path and os.path.exists(seg_mask_path):
                                    try:
                                        raw_seg = np.array(Image.open(seg_mask_path))
                                        if raw_seg.ndim == 3:
                                            raw_seg = raw_seg[:, :, 0]
                                        if raw_seg.max() > 0:
                                            import cv2
                                            scaled_seg = (raw_seg.astype(np.float32) / float(raw_seg.max()) * 255.0).astype(np.uint8)
                                            color_seg  = cv2.applyColorMap(scaled_seg, cv2.COLORMAP_TURBO)
                                            seg_map_pil = Image.fromarray(cv2.cvtColor(color_seg, cv2.COLOR_BGR2RGB)).resize(img_size)
                                        else:
                                            seg_map_pil = Image.fromarray(raw_seg).convert('RGB').resize(img_size)
                                    except Exception:
                                        seg_map_pil = Image.new('RGB', img_size, (0, 0, 0))

                            if seg_map_pil is None:
                                seg_map_pil = Image.new('RGB', img_size, (0, 0, 0))

                            panels       = [gt_img_pil, gt_mask_pil, seg_map_pil, full_path_pil,
                                            hole_path_pil, hole_lines_pil,
                                            da_offset_pil, pred_img_pil, pred_mask_pil]
                            panel_labels = ['GT', 'Masked Input', 'Segment Map', 'Full Path',
                                            'Hole Heatmap', 'Hole Lines',
                                            'DA Offsets', 'Raw Pred', 'Merged']
                            total_width = sum(p.size[0] for p in panels)
                            max_height  = max(p.size[1] for p in panels)
                            label_h     = 20
                            new_im      = Image.new('RGB', (total_width, max_height + label_h), (20, 20, 20))
                            from PIL import ImageDraw
                            draw_im = ImageDraw.Draw(new_im)
                            x_off = 0
                            for im, lbl in zip(panels, panel_labels):
                                new_im.paste(im, (x_off, 0))
                                draw_im.text((x_off + 4, max_height + 2), lbl, fill=(220, 220, 220))
                                x_off += im.size[0]

                            name      = f"{base_name}_iter{iteration}.png"
                            save_path = os.path.join(path_val, name)
                            new_im.save(save_path)
                            print(f"Saved validation image {val_count+1}/{len(all_indices)} to {save_path}")
                            val_count += 1

                        if self.config.RANK == 0 and len(val_psnr_list) > 0:
                            mean_val_psnr = float(np.mean(val_psnr_list))
                            print(f"\n[VALIDATION] Iteration {iteration} | Average Validation PSNR: {mean_val_psnr:.2f} dB")

                            # 1. Save best model per epoch
                            if mean_val_psnr > self.best_epoch_psnr:
                                self.best_epoch_psnr = mean_val_psnr
                                epoch_prefix = f"InpaintingModel_best_epoch_{epoch}"
                                self.inpaint_model.save(prefix=epoch_prefix)
                                print(f"[BEST MODEL] Epoch {epoch} new best PSNR ({mean_val_psnr:.2f} dB) -> Saved {epoch_prefix}_gen.pth")

                            # 2. Keep top 5 best models overall
                            if len(self.top5_best_models) < 5 or mean_val_psnr > self.top5_best_models[-1]['psnr']:
                                top_prefix = f"InpaintingModel_top_psnr_{mean_val_psnr:.2f}_iter_{iteration}"
                                self.inpaint_model.save(prefix=top_prefix)
                                gen_path = os.path.join(self.config.PATH, f"{top_prefix}_gen.pth")
                                dis_path = os.path.join(self.config.PATH, f"{top_prefix}_dis.pth")
                                
                                self.top5_best_models.append({
                                    'psnr': mean_val_psnr,
                                    'epoch': epoch,
                                    'iter': iteration,
                                    'prefix': top_prefix,
                                    'gen': gen_path,
                                    'dis': dis_path
                                })
                                self.top5_best_models.sort(key=lambda x: x['psnr'], reverse=True)

                                if len(self.top5_best_models) > 5:
                                    removed = self.top5_best_models.pop()
                                    for pth in [removed['gen'], removed['dis']]:
                                        if os.path.exists(pth):
                                            try:
                                                os.remove(pth)
                                            except Exception as e:
                                                print(f"Warning: Failed to remove old top-5 checkpoint {pth}: {e}")
                                top_scores = [f"{m['psnr']:.2f}dB" for m in self.top5_best_models]
                                print(f"[TOP 5 MODELS] Updated Top 5 models list: {top_scores}")

                        self.inpaint_model.train()
                    ##############


                    # log model at checkpoints
                    if self.config.RANK == 0 and self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                        self.log(logs)



                    # save model at checkpoints and upload to C2 server
                    if self.config.RANK == 0 and self.config.SAVE_INTERVAL != 0 and iteration % self.config.SAVE_INTERVAL == 0:
                        self.save()
                        # Persist epoch so process restarts resume from the right epoch
                        with open(self.epoch_state_file, 'w') as _ef:
                            json.dump({'epoch': epoch, 'iteration': iteration}, _ef)
        except (KeyboardInterrupt, SystemExit):
            print("\n[Training Interrupted by User (Ctrl+C)] Cleaning up and exiting...")
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
            import sys
            sys.exit(0)

        print('\nEnd training....')


    def test(self):

        self.inpaint_model.eval()
        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
        )
        
        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []
        
        print('here')
        index = 0
        for items in test_loader:
            images, masks = self.cuda(*items)
            index += 1

            # inpaint model
            if model == 2:
                

                inputs = (images * (1 - masks))
                with torch.inference_mode():             
                    with torch.cuda.amp.autocast():
                        outputs_img = self.inpaint_model(images, masks)

                outputs_img = outputs_img.float()
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))
                
                print('outpus_size', outputs_merged.size())
                print('images', images.size())
                
                
                
                psnr, ssim = self.metric(images, outputs_merged)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                
                if torch.cuda.is_available():
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(), self.transf(images[0].cpu()).cuda()).item()
                    lpips_list.append(pl)
                else:
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(images[0].cpu())).item()
                    lpips_list.append(pl)                
                
                l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
                l1_list.append(l1_loss)

                print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
                                                                                ssim, np.average(ssim_list),
                                                                                l1_loss, np.average(l1_list),
                                                                                pl, np.average(lpips_list),
                                                                                len(ssim_list)))

                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(inputs),
                    self.postprocess(outputs_img),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                path_masked = os.path.join(self.results_path,self.model_name,'masked_lama')
                path_result = os.path.join(self.results_path, self.model_name,'result_lama')
                path_joint = os.path.join(self.results_path,self.model_name,'joint_lama')

                name = self.test_dataset.load_name(index-1)[:-4]+'.png'

                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)
                

                masked_images = self.postprocess(images*(1-masks)+masks)[0]
                images_result = self.postprocess(outputs_merged)[0]

                print(os.path.join(path_joint,name[:-4]+'.png'))

                images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                imsave(masked_images,os.path.join(path_masked,name))
                imsave(images_result,os.path.join(path_result,name))

                print(name + ' complete!')

            # inpaint with joint model
        print('\nEnd Testing')
        
        print('edge_psnr_ave:{} edge_ssim_ave:{} l1_ave:{} lpips:{}'.format(np.average(psnr_list),
                                                                                 np.average(ssim_list),
                                                                                 np.average(l1_list),
                                                                                 np.average(lpips_list)))



    def log(self, logs):
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))

        ssim = compare_ssim(gt, pre, multichannel=True, channel_axis=-1, data_range=255)

        return psnr, ssim
