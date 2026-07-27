import os
import sys
import json
import glob
import time
import requests
import torchvision
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import lpips
from .models import InpaintingModel
from .dataset import Dataset
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
try:
    import wandb
except ImportError:
    wandb = None

C2_SERVER_URL = "http://localhost:5000"
C2_SESSION = "SEM-Net-Run"

def upload_artifact(filename, file_path):
    try:
        url = f"{C2_SERVER_URL}/api/upload_artifact"
        files = {'file': open(file_path, 'rb')}
        data = {'session': C2_SESSION, 'filename': filename}
        r = requests.post(url, data=data, files=files, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"[C2 UPLOAD] Failed to upload {filename}: {e}")
        return False

def upload_artifact_chunked(filename, file_path, chunk_size=25*1024*1024):
    try:
        if not os.path.exists(file_path): return False
        total_size = os.path.getsize(file_path)
        total_chunks = (total_size + chunk_size - 1) // chunk_size
        print(f"[C2 UPLOAD] Starting chunked upload for {filename} ({total_size/(1024*1024):.1f} MB, {total_chunks} chunks)...")
        with open(file_path, 'rb') as f:
            for i in range(total_chunks):
                chunk_data = f.read(chunk_size)
                files = {'chunk': (filename, chunk_data)}
                data = {
                    'session': C2_SESSION,
                    'filename': filename,
                    'chunk_index': i,
                    'total_chunks': total_chunks
                }
                res = requests.post(f"{C2_SERVER_URL}/api/upload_artifact_chunk", data=data, files=files, timeout=60)
                if res.status_code != 200:
                    status_info = f"Status: {res.status_code}, Response: {res.text[:300]}" if 'res' in locals() else "No response"
                    print(f"[C2 UPLOAD] Failed to upload chunk {i} ({status_info}). Aborting.")
                    return False
        print(f"[C2 UPLOAD] Successfully uploaded {filename}!")
        return True
    except Exception as e:
        print(f"[C2 UPLOAD] Error uploading {filename}: {e}")
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
                _ = models.vgg19(pretrained=True)
                _ = lpips.LPIPS(net='vgg')
            dist.barrier()

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        # datasets
        if self.config.MODEL == 2:
            train_seg = getattr(config, 'TRAIN_SEGMENT_FLIST', None)
            test_seg = getattr(config, 'TEST_SEGMENT_FLIST', None)
            self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, train_seg, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, test_seg, config.TEST_MASK_FLIST, augment=False, training=False)

        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')
        self.epoch_state_file = os.path.join(config.PATH, 'epoch_state.json')

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
            batch_size=max(1, self.config.BATCH_SIZE // self.config.WORLD_SIZE),
            num_workers=6,
            drop_last=True,
            shuffle=(sampler is None),
            pin_memory=True,
            sampler=sampler
        )

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
        
        _METRIC_KEYS = ['gen_loss', 'dis_loss', 'l1_loss', 'perceptual_loss',
                        'style_loss', 'sym_loss', 'gan_loss', 'psnr', 'mae']
        _metric_buf = {k: [] for k in _METRIC_KEYS}
        _metric_buf_epoch = []
        
        while(keep_training):
            epoch += 1
            if self.config.RANK == 0:
                print(f"Training epoch: {epoch}")
                progbar = Progbar(len(train_loader), width=20, stateful_metrics=['epoch', 'iter'])
            
            if sampler is not None:
                sampler.set_epoch(epoch)
            for items in train_loader:
                iteration = self.inpaint_model.iteration
                self.inpaint_model.train()

                if model == 2:
                    if len(items) == 3:
                        images, masks, segment_maps = self.cuda(*items)
                    else:
                        images, masks = self.cuda(*items)
                        segment_maps = None

                    outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, gen_symmetry_loss = self.inpaint_model.process(images,masks)
                    outputs_merged = (outputs_img * masks) + (images * (1-masks))

                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    self.inpaint_model.backward(gen_loss, dis_loss)

                    if self.config.RANK == 0 and self.inpaint_model.iteration == 1:
                        print("\n=== nvidia-smi (GPU allocation after 1st iteration) ===")
                        import subprocess
                        try:
                            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                            print(res.stdout)
                        except Exception as e:
                            print(f"Could not run nvidia-smi: {e}")
                        print("=========================================================\n")

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

                    if iteration > 0 and iteration % 300 == 0:
                        try:
                            n = len(_metric_buf['psnr'])
                            all_metrics_payload = {
                                "iteration": iteration,
                                "epoch": round(sum(_metric_buf_epoch) / len(_metric_buf_epoch), 2),
                                "_samples": n,
                                "session": C2_SESSION
                            }
                            for k in _METRIC_KEYS:
                                vals = _metric_buf[k]
                                all_metrics_payload[k] = round(sum(vals) / len(vals), 6) if vals else 0.0
                            requests.post(f"{C2_SERVER_URL}/api/all_metrics", json=all_metrics_payload, timeout=2)
                        except Exception:
                            pass
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

                if iteration % 50 == 0:
                    try:
                        res = requests.get(f"{C2_SERVER_URL}/api/command", params={"session": C2_SESSION}, timeout=2)
                        if res.status_code == 200:
                            cmd_data = res.json()
                            cmd = cmd_data.get('command', 'run')
                            if cmd == 'stop':
                                print("\nC2 Server requested STOP. Halting gracefully.")
                                keep_training = False
                                break
                            elif cmd == 'restart_pull':
                                print("\nC2 Server requested RESTART_PULL. Exiting 42.")
                                sys.exit(42)

                        res_shell = requests.get(f"{C2_SERVER_URL}/api/pop_shell_command", params={"session": C2_SESSION}, timeout=2)
                        if res_shell.status_code == 200:
                            shell_cmd = res_shell.json().get('shell_command')
                            if shell_cmd:
                                import subprocess
                                print(f"\n[C2 REMOTE COMMAND] Executing: {shell_cmd}")
                                try:
                                    result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, timeout=30)
                                    if result.stdout: print(result.stdout)
                                    if result.stderr: print(result.stderr)
                                    print(f"[C2 REMOTE COMMAND] Exit code: {result.returncode}\n")
                                except Exception as e:
                                    print(f"[C2 REMOTE COMMAND] Error: {str(e)}\n")
                    except Exception:
                        pass

                eval_interval = getattr(self.config, 'EVAL_INTERVAL', 100)
                if eval_interval > 0 and iteration % eval_interval == 0:
                    create_dir(self.results_path)
                    path_val = os.path.join(self.results_path, self.model_name, 'validation')
                    create_dir(path_val)
                    
                    self.inpaint_model.eval()
                    import matplotlib.pyplot as plt
                    import io
                    import numpy as np
                    from torch.utils.data import Subset

                    n_test      = len(self.test_dataset)
                    first_idx   = list(range(min(5, n_test)))
                    last_idx    = list(range(max(0, n_test - 5), n_test))
                    all_indices = list(dict.fromkeys(first_idx + last_idx))
                    val_loader  = DataLoader(dataset=Subset(self.test_dataset, all_indices),
                                             batch_size=1, num_workers=0, shuffle=False)

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

                    def _draw_hole_heatmap(scan_orders, mask_np, img_size, patch_size):
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
                            t = local_rank / max(n - 1, 1)
                            r, g, b, _ = cmap_plasma(t)
                            color = (int(r*255), int(g*255), int(b*255))
                            y0, x0 = int(p_i*patch_size), int(p_j*patch_size)
                            y1, x1 = min(y0+patch_size, H_px), min(x0+patch_size, W_px)
                            canvas[y0:y1, x0:x1] = color

                        return Image.fromarray(canvas).resize(img_size)

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
                            t = local_rank / max(n - 1, 1)
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
                    for val_items in val_loader:
                        if len(val_items) == 3:
                            val_images, val_masks, val_segments = self.cuda(*val_items)
                        else:
                            val_images, val_masks = self.cuda(*val_items)
                            val_segments = None

                        val_inputs = (val_images * (1 - val_masks)) + val_masks
                        with torch.no_grad():
                            val_outputs_img = self.inpaint_model(val_images, val_masks)
                        
                        val_outputs_merged = (val_outputs_img * val_masks) + (val_images * (1 - val_masks))

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
                            print(f"Could not extract scan_orders: {e}")

                        gt_img_pil    = Image.fromarray(self.postprocess(val_images)[0].cpu().numpy().astype(np.uint8))
                        gt_mask_pil   = Image.fromarray(self.postprocess(val_inputs)[0].cpu().numpy().astype(np.uint8))
                        if val_segments is not None:
                            seg_img_pil = Image.fromarray(self.postprocess(val_segments)[0].cpu().numpy().astype(np.uint8))
                        else:
                            seg_img_pil = Image.new('RGB', gt_img_pil.size, (40, 60, 90))

                        pred_img_pil  = Image.fromarray(self.postprocess(val_outputs_img)[0].cpu().numpy().astype(np.uint8))
                        pred_mask_pil = Image.fromarray(self.postprocess(val_outputs_merged)[0].cpu().numpy().astype(np.uint8))
                        img_size = gt_img_pil.size
                        mask_np  = val_masks[0, 0].cpu().float().numpy()

                        full_path_pil = _draw_path_panel(scan_orders, mask_np, gt_mask_pil,
                                                         patch_size, img_size)

                        hole_path_pil = _draw_hole_heatmap(scan_orders, mask_np, img_size, patch_size)

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

                        # ── 9-panel stitch ────────────────────────────────────────
                        panels       = [gt_img_pil, gt_mask_pil, seg_img_pil, full_path_pil,
                                        hole_path_pil, hole_lines_pil,
                                        da_offset_pil, pred_img_pil, pred_mask_pil]
                        panel_labels = ['GT', 'Masked Input', 'SAM Segment Map', 'Full Path',
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

                        orig_idx  = all_indices[val_count]
                        name      = self.test_dataset.load_name(orig_idx)[:-4] + f'_iter{iteration}.png'
                        save_path = os.path.join(path_val, name)
                        new_im.save(save_path)
                        print(f"Saved validation image {val_count+1}/{len(all_indices)} to {save_path}")
                        val_count += 1

                    if self.config.RANK == 0:
                        print(f"[VAL] Evaluation output completed for iteration {iteration}!")

                if self.config.RANK == 0 and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
                    if os.path.exists(self.epoch_state_file):
                        try:
                            with open(self.epoch_state_file, 'w') as _ef:
                                json.dump({'epoch': epoch, 'iteration': iteration}, _ef)
                        except Exception:
                            pass

    def eval(self):
        val_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            drop_last=False,
            shuffle=False
        )

        model = self.config.MODEL
        total = len(self.test_dataset)

        if self.config.RANK == 0:
            progbar = Progbar(total, width=20, stateful_metrics=['it'])
        index = 0

        path = os.path.join(self.results_path, self.model_name, 'eval')
        create_dir(path)

        self.inpaint_model.eval()

        for items in val_loader:
            index += 1
            if len(items) == 3:
                images, masks, segment_maps = self.cuda(*items)
            else:
                images, masks = self.cuda(*items)

            outputs_img = self.inpaint_model(images, masks)
            outputs_merged = (outputs_img * masks) + (images * (1 - masks))

            name = self.test_dataset.load_name(index - 1)
            images = self.postprocess(images)[0]
            masks = self.postprocess(masks)[0]
            outputs_img = self.postprocess(outputs_img)[0]
            outputs_merged = self.postprocess(outputs_merged)[0]

            imsave(outputs_merged, os.path.join(path, name))

            if self.config.RANK == 0:
                progbar.add(1)

    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]

    def postprocess(self, img):
        img = (img + 1) / 2 * 255.0
        return img.clamp(0, 255)
