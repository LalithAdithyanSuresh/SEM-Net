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
                    import matplotlib.pyplot as plt
                    import io
                    import numpy as np
                    from torch.utils.data import Subset

                    # 5 from start + 5 from end of test dataset
                    n_test      = len(self.test_dataset)
                    first_idx   = list(range(min(5, n_test)))
                    last_idx    = list(range(max(0, n_test - 5), n_test))
                    all_indices = list(dict.fromkeys(first_idx + last_idx))
                    val_loader  = DataLoader(dataset=Subset(self.test_dataset, all_indices),
                                             batch_size=1, num_workers=0, shuffle=False)

                    # ── Helper: render one scan-path panel ────────────────────
                    def _draw_path_panel(scan_orders, mask_np, bg_pil, patch_size, img_size, hole_only=False):
                        from matplotlib.collections import LineCollection
                        fig = plt.figure(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
                        ax  = fig.add_axes([0, 0, 1, 1])
                        ax.axis('off')
                        if hole_only:
                            ax.imshow(np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8))
                        else:
                            ax.imshow(np.array(bg_pil))
                        if scan_orders:
                            H_m, W_m = mask_np.shape
                            y_all = np.array([r * patch_size + patch_size/2.0 for r, c in scan_orders])
                            x_all = np.array([c * patch_size + patch_size/2.0 for r, c in scan_orders])
                            if hole_only:
                                keep = [k for k, (r, c) in enumerate(scan_orders)
                                        if mask_np[min(int(r*patch_size+patch_size//2), H_m-1),
                                                   min(int(c*patch_size+patch_size//2), W_m-1)] > 0.5]
                                if not keep:
                                    plt.close(fig)
                                    return Image.new('RGB', img_size, (30, 30, 30))
                                y_coords, x_coords = y_all[keep], x_all[keep]
                            else:
                                y_coords, x_coords = y_all, x_all
                            pts  = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
                            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                            lc   = LineCollection(segs, cmap='rainbow',
                                                  norm=plt.Normalize(0, len(x_coords)),
                                                  alpha=0.80, linewidths=1.5)
                            lc.set_array(np.arange(len(x_coords)))
                            ax.add_collection(lc)
                            if len(x_coords) > 0:
                                ax.scatter([x_coords[0]],  [y_coords[0]],  color='lime', s=45, zorder=5, edgecolors='black')
                                ax.scatter([x_coords[-1]], [y_coords[-1]], color='red',  s=45, zorder=5, edgecolors='black')
                                step = max(1, len(x_coords) // 15)
                                for k in range(0, len(x_coords)-1, step):
                                    ddx = x_coords[k+1]-x_coords[k]; ddy = y_coords[k+1]-y_coords[k]
                                    dist = np.hypot(ddx, ddy)
                                    if dist > 0:
                                        ax.arrow(x_coords[k], y_coords[k],
                                                 (ddx/dist)*(patch_size*0.45), (ddy/dist)*(patch_size*0.45),
                                                 color='white', head_width=patch_size*0.4, alpha=1.0, zorder=6)
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=100)
                        plt.close(fig)
                        buf.seek(0)
                        return Image.open(buf).convert('RGB').resize(img_size)

                    val_count = 0
                    for val_items in val_loader:
                        val_images, val_masks = self.cuda(*val_items)
                        val_inputs = (val_images * (1 - val_masks)) + val_masks
                        with torch.no_grad():
                            val_outputs_img = self.inpaint_model(val_images, val_masks)
                        
                        val_outputs_merged = (val_outputs_img * val_masks) + (val_images * (1 - val_masks))

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
                            print(f"Could not extract scan_orders: {e}")

                        # ── PIL conversions ───────────────────────────────────────
                        gt_img_pil    = Image.fromarray(self.postprocess(val_images)[0].cpu().numpy().astype(np.uint8))
                        gt_mask_pil   = Image.fromarray(self.postprocess(val_inputs)[0].cpu().numpy().astype(np.uint8))
                        pred_img_pil  = Image.fromarray(self.postprocess(val_outputs_img)[0].cpu().numpy().astype(np.uint8))
                        pred_mask_pil = Image.fromarray(self.postprocess(val_outputs_merged)[0].cpu().numpy().astype(np.uint8))
                        img_size = gt_img_pil.size
                        mask_np  = val_masks[0, 0].cpu().float().numpy()  # H×W, 1=hole

                        # ── Panel 3: Full path (all tokens on masked-input bg) ────
                        full_path_pil = _draw_path_panel(scan_orders, mask_np, gt_mask_pil,
                                                         patch_size, img_size, hole_only=False)

                        # ── Panel 4: Hole-only path (black bg, only masked patches)
                        hole_path_pil = _draw_path_panel(scan_orders, mask_np, gt_mask_pil,
                                                         patch_size, img_size, hole_only=True)

                        # ── Panel 5: DA-Mamba offset heatmap (every image) ────────
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

                        # ── 7-panel stitch ────────────────────────────────────────
                        panels       = [gt_img_pil, gt_mask_pil, full_path_pil, hole_path_pil,
                                        da_offset_pil, pred_img_pil, pred_mask_pil]
                        panel_labels = ['GT', 'Masked Input', 'Full Path', 'Hole Path',
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

                        # ── Save & upload ─────────────────────────────────────────
                        orig_idx  = all_indices[val_count]
                        name      = self.test_dataset.load_name(orig_idx)[:-4] + f'_iter{iteration}.png'
                        save_path = os.path.join(path_val, name)
                        new_im.save(save_path)
                        print(f"Saved validation image {val_count+1}/{len(all_indices)} to {save_path}")
                        try:
                            with open(save_path, 'rb') as f:
                                requests.post(f"{C2_SERVER_URL}/api/upload_image",
                                              files={'file': (name, f, 'image/png')}, timeout=5)
                        except Exception:
                            pass
                        val_count += 1

                    # ── C2 metrics snapshot ───────────────────────────────────────
                    try:
                        requests.post(f"{C2_SERVER_URL}/api/metrics", timeout=5, json={
                            "epoch": epoch, "iteration": iteration,
                            "val_gen_l1": float(gen_l1_loss),
                            "val_gen_pl":  float(gen_content_loss),
                            "val_gen_adv": float(gen_gan_loss),
                        })
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
