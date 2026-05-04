import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

# Assume the C2 URL is passed via environment variable (or default to port 443 of VPS)
C2_SERVER_URL = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')

class sem():
    def __init__(self, config):
        self.config = config


        if config.MODEL == 2:
            model_name = 'inpaint'

        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
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

    def load(self):


        if self.config.MODEL == 2:
            self.inpaint_model.load()


    def save(self):
 
        if self.config.MODEL == 2:
            self.inpaint_model.save()


    def train(self):
        
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
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
        
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)


            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            

            for items in train_loader:
                iteration = self.inpaint_model.iteration  # read BEFORE process() increments it
                self.inpaint_model.train()


                if model == 2:
                    images, masks = self.cuda(*items)

                    outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, gen_symmetry_loss = self.inpaint_model.process(images,masks)
                    outputs_merged = (outputs_img * masks) + (images * (1-masks))

                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    self.inpaint_model.backward(gen_loss, dis_loss)

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

                    # --- C2: send TRUE 300-iteration average once per 300 iters ---
                    if iteration > 0 and iteration % 300 == 0:
                        try:
                            n = len(_metric_buf['psnr'])
                            all_metrics_payload = {
                                "iteration": iteration,
                                "epoch": round(sum(_metric_buf_epoch) / len(_metric_buf_epoch), 2),
                                "_samples": n,  # how many iterations this average covers
                            }
                            for k in _METRIC_KEYS:
                                vals = _metric_buf[k]
                                all_metrics_payload[k] = round(sum(vals) / len(vals), 6) if vals else 0.0
                            requests.post(f"{C2_SERVER_URL}/api/all_metrics", json=all_metrics_payload, timeout=2)
                        except Exception:
                            pass
                        # Reset buffers for the next 300-iter window
                        _metric_buf = {k: [] for k in _METRIC_KEYS}
                        _metric_buf_epoch = []
                    iteration = self.inpaint_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                if iteration % 10 == 0 and wandb is not None and wandb.run is not None:
                        wandb.log({'gen_loss': gen_loss, 'l1_loss': gen_l1_loss, 'style_loss': gen_style_loss,
                                   'perceptual loss': gen_content_loss, 'gen_gan_loss': gen_gan_loss,
                                   'gen_symmetry_loss': gen_symmetry_loss,
                                   'dis_loss': dis_loss}, step=iteration)
		 
                # ---- C2 COMMAND POLLING (Less frequent for speed) ----
                if iteration % 50 == 0:
                    try:
                        # 1. Fetch training command (stop/run/etc)
                        res = requests.get(f"{C2_SERVER_URL}/api/command", timeout=2)
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

                        # 2. Fetch custom shell commands (dedicated endpoint to avoid race conditions)
                        res_shell = requests.get(f"{C2_SERVER_URL}/api/pop_shell_command", timeout=2)
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
                        pass # Ignore net errors

                if iteration % 1000 == 0:
                    create_dir(self.results_path)
                    path_val = os.path.join(self.results_path, self.model_name, 'validation')
                    create_dir(path_val)
                    
                    self.inpaint_model.eval()
                    val_loader = DataLoader(dataset=self.test_dataset, batch_size=1, num_workers=0, shuffle=False)
                    import matplotlib.pyplot as plt
                    import io
                    import numpy as np

                    val_count = 0
                    for val_items in val_loader:
                        if val_count >= 5: break
                        val_images, val_masks = self.cuda(*val_items)
                        val_inputs = (val_images * (1 - val_masks)) + val_masks
                        with torch.no_grad():
                            val_outputs_img = self.inpaint_model(val_images, val_masks)
                        
                        val_outputs_merged = (val_outputs_img * val_masks) + (val_images * (1 - val_masks))
                        
                        # Extract the path from the first CombinedAdaptiveMambaLayer instance
                        # Assuming it's in encoder_level1[0].attn
                        patch_size = 8
                        try:
                            if hasattr(self.inpaint_model.generator, 'module'):
                                attn_layer = self.inpaint_model.generator.module.encoder_level1[0].attn
                            else:
                                attn_layer = self.inpaint_model.generator.encoder_level1[0].attn
                            
                            # Use new tensor checking (tensor is not None instead of boolean implicitly checking truth value)
                            if getattr(attn_layer, 'last_scan_orders', None) is not None:
                                scan_orders_tensor = attn_layer.last_scan_orders[0] # 1D tensor from Batch 0
                                patch_size = getattr(attn_layer, 'last_patch_size', 8)
                                W_p = getattr(attn_layer, 'last_W_p', 256//patch_size)
                                
                                # Convert linear indices to (p_i, p_j) tuples
                                scan_orders = []
                                for idx in scan_orders_tensor.cpu().tolist():
                                    p_i = idx // W_p
                                    p_j = idx % W_p
                                    scan_orders.append((p_i, p_j))
                            else:
                                scan_orders = None
                        except Exception as e:
                            print(f"Could not extract scan_orders: {e}")
                            scan_orders = None

                        # Convert tensors to PIL images
                        gt_img_pil = Image.fromarray(self.postprocess(val_images)[0].cpu().numpy().astype(np.uint8))
                        gt_mask_pil = Image.fromarray(self.postprocess(val_inputs)[0].cpu().numpy().astype(np.uint8))
                        pred_img_pil = Image.fromarray(self.postprocess(val_outputs_img)[0].cpu().numpy().astype(np.uint8))
                        pred_mask_pil = Image.fromarray(self.postprocess(val_outputs_merged)[0].cpu().numpy().astype(np.uint8))

                        # Draw path (ONLY for first image for speed)
                        if val_count == 0:
                            import matplotlib.pyplot as plt
                            import io
                            fig = plt.figure(figsize=(gt_mask_pil.size[0]/100, gt_mask_pil.size[1]/100), dpi=100)
                            ax = fig.add_axes([0, 0, 1, 1])
                            ax.axis('off')
                            ax.imshow(gt_mask_pil)
                            if scan_orders is not None:
                                from matplotlib.collections import LineCollection
                                y_coords = np.array([p_i * patch_size + patch_size/2.0 for p_i, p_j in scan_orders])
                                x_coords = np.array([p_j * patch_size + patch_size/2.0 for p_i, p_j in scan_orders])
                                points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
                                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                                norm = plt.Normalize(0, len(x_coords))
                                lc = LineCollection(segments, cmap='rainbow', norm=norm, alpha=0.75, linewidths=1.5)
                                lc.set_array(np.arange(len(x_coords)))
                                ax.add_collection(lc)
                                if len(x_coords) > 0:
                                    ax.scatter([x_coords[0]], [y_coords[0]], color='lime', s=45, zorder=5, edgecolors='black', label='Start')
                                    ax.scatter([x_coords[-1]], [y_coords[-1]], color='red', s=45, zorder=5, edgecolors='black', label='End')
                                    step = max(1, len(x_coords) // 15)
                                    for i in range(0, len(x_coords)-1, step):
                                        dx = x_coords[i+1] - x_coords[i]; dy = y_coords[i+1] - y_coords[i]
                                        dist = np.hypot(dx, dy)
                                        if dist > 0:
                                            ax.arrow(x_coords[i], y_coords[i], (dx/dist)*(patch_size*0.45), (dy/dist)*(patch_size*0.45), 
                                                     color='white', head_width=patch_size*0.4, alpha=1.0, zorder=6)
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', dpi=100)
                            plt.close(fig)
                            buf.seek(0)
                            gt_mask_paths_pil = Image.open(buf).convert('RGB')
                            gt_mask_paths_pil = gt_mask_paths_pil.resize(gt_mask_pil.size)

                            # ---- Panel 4: DA-Mamba Offset Heatmap ----
                            try:
                                import cv2
                                da_offset_pil = None
                                da_offset_map = getattr(attn_layer.da_scan, 'last_offset_map', None)
                                if da_offset_map is not None:
                                    off = da_offset_map[0].cpu().float().numpy()
                                    G = off.shape[-1] // 2
                                    dx = off[..., :G]; dy = off[..., G:]
                                    mag = np.sqrt(dx**2 + dy**2).mean(axis=-1)
                                    mag = (mag - mag.min()) / (mag.max() + 1e-8)
                                    mag_u8 = (mag * 255).astype(np.uint8)
                                    mag_bgr = cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)
                                    mag_rgb = cv2.cvtColor(mag_bgr, cv2.COLOR_BGR2RGB)
                                    da_offset_pil = Image.fromarray(mag_rgb).resize(gt_img_pil.size)
                                if da_offset_pil is None:
                                    da_offset_pil = Image.new('RGB', gt_img_pil.size, (80, 80, 80))
                            except Exception:
                                da_offset_pil = Image.new('RGB', gt_img_pil.size, (80, 80, 80))
                        else:
                            # Skip heavy viz for images 1-4 to save ~5-10 seconds per iteration
                            gt_mask_paths_pil = Image.new('RGB', gt_img_pil.size, (40, 40, 40))
                            da_offset_pil = Image.new('RGB', gt_img_pil.size, (40, 40, 40))

                        # Concatenate 6 panels horizontally
                        panels = [gt_img_pil, gt_mask_pil, gt_mask_paths_pil, da_offset_pil, pred_img_pil, pred_mask_pil]
                        panel_labels = ['GT', 'Masked Input', 'VAMamba Path', 'DA-Mamba Offsets', 'Raw Pred', 'Merged']
                        widths, heights = zip(*(i.size for i in panels))
                        total_width = sum(widths)
                        max_height = max(heights)
                        
                        # Add label bar (20px) below each panel
                        label_h = 20
                        new_im = Image.new('RGB', (total_width, max_height + label_h), (20, 20, 20))
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(new_im)
                        x_offset = 0
                        for im, lbl in zip(panels, panel_labels):
                            new_im.paste(im, (x_offset, 0))
                            # Draw label
                            draw.text((x_offset + 4, max_height + 2), lbl, fill=(220, 220, 220))
                            x_offset += im.size[0]

                        # Save stitched image
                        name = self.test_dataset.load_name(val_count)[:-4] + f'_iter{iteration}.png'
                        
                        save_path = os.path.join(path_val, name)
                        new_im.save(save_path)
                        print(f"Saved validation image {val_count+1}/10 to {save_path}")
                        
                        # ---- C2 IMAGE UPLOAD ----
                        try:
                            with open(save_path, 'rb') as f:
                                files = {'file': (name, f, 'image/png')}
                                requests.post(f"{C2_SERVER_URL}/api/upload_image", files=files, timeout=5)
                        except Exception:
                            pass
                        val_count += 1

                    # ---- C2 METRICS UPLOAD ----
                    try:
                        metrics_payload = {
                            "epoch": epoch,
                            "iteration": iteration,
                            "val_gen_l1": float(gen_l1_loss),
                            "val_gen_pl": float(gen_content_loss),
                            "val_gen_adv": float(gen_gan_loss)
                        }
                        requests.post(f"{C2_SERVER_URL}/api/metrics", json=metrics_payload, timeout=5)
                    except Exception:
                        pass

                    self.inpaint_model.train()
                ##############


                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)



                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
                    # Persist epoch so process restarts resume from the right epoch
                    with open(self.epoch_state_file, 'w') as _ef:
                        json.dump({'epoch': epoch, 'iteration': iteration}, _ef)
        print('\nEnd training....')


    def test(self):

        self.inpaint_model.eval()
        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
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
                with torch.no_grad():             

                    outputs_img = self.inpaint_model(images, masks)

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
