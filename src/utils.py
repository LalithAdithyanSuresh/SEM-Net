import torch
import os
import cv2
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)




class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s - %.2fs/it' % (eta_format, time_per_unit)
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    # Ensure we are dealing with numbers, not tensors
                    val0 = self._values[k][0]
                    val1 = self._values[k][1]
                    if hasattr(val0, 'item'): val0 = val0.item()
                    if hasattr(val1, 'item'): val1 = val1.item()
                    avg = np.mean(val0 / max(1, val1))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


def PositionalEncoding(d_model, max_len=5000):
    import math
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def prepare_tmp_dir(config, tmp_dir, is_training=False, selected_images=None, output_dir=None):
    import shutil
    import glob
    from concurrent.futures import ThreadPoolExecutor
    
    # Ensure tmp_dir exists
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    
    print(f"Preparing datasets and directories in fast storage: {tmp_dir}")
    
    # 1. Output/checkpoint directory mapping
    mapped_output = None
    if output_dir is not None:
        mapped_output = os.path.join(tmp_dir, os.path.basename(output_dir.rstrip('/\\')))
        print(f"Mapping output directory: {output_dir} -> {mapped_output}")
        create_dir(mapped_output)
    
    orig_path = config.PATH
    config.PATH = os.path.join(tmp_dir, os.path.basename(orig_path.rstrip('/\\')))
    print(f"Mapping checkpoints/logs directory: {orig_path} -> {config.PATH}")
    create_dir(config.PATH)
    
    # Copy checkpoints if they exist (generator and discriminator)
    for model_file in ['InpaintingModel_gen.pth', 'InpaintingModel_dis.pth', 'config.yml']:
        src_file = os.path.join(orig_path, model_file)
        if os.path.exists(src_file):
            dst_file = os.path.join(config.PATH, model_file)
            if not os.path.exists(dst_file) or os.path.getsize(src_file) != os.path.getsize(dst_file):
                print(f"Copying checkpoint {model_file} to local storage...")
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    print(f"Failed to copy checkpoint {model_file}: {e}")
            
    # 2. Helper to copy files concurrently
    def copy_file_if_missing(src, dst):
        if not os.path.exists(src):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst) or os.path.getsize(src) != os.path.getsize(dst):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                pass

    def to_rel(path):
        if os.path.isabs(path):
            try:
                return os.path.relpath(path)
            except Exception:
                return path.lstrip('/\\').replace(':', '')
        return path

    def copy_dir_contents(src_dir, dst_dir, limit_files=None):
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} does not exist.")
            return
        os.makedirs(dst_dir, exist_ok=True)
        
        # Get all files recursively
        all_files = glob.glob(os.path.join(src_dir, '**', '*'), recursive=True)
        all_files = [f for f in all_files if os.path.isfile(f)]
        
        if limit_files is not None:
            files_to_copy = [f for f in all_files if os.path.basename(f) in limit_files]
        else:
            files_to_copy = all_files
            
        # Pre-filter to only copy missing or size-mismatched files
        needed_files = []
        for src_file in files_to_copy:
            rel_path = os.path.relpath(src_file, src_dir)
            dst_file = os.path.join(dst_dir, rel_path)
            if not os.path.exists(dst_file) or os.path.getsize(src_file) != os.path.getsize(dst_file):
                needed_files.append((src_file, dst_file))
                
        if not needed_files:
            print(f"All files for {dst_dir} are already present. Skipping copy.")
            return
            
        print(f"Copying {len(needed_files)} missing/modified files from {src_dir} to {dst_dir}...")
        
        with ThreadPoolExecutor(max_workers=16) as copypool:
            futures = []
            for src_file, dst_file in needed_files:
                futures.append(copypool.submit(copy_file_if_missing, src_file, dst_file))
            for fut in futures:
                fut.result()
        print(f"Copying complete for {dst_dir}")

    # 3. Copy/map mask datasets
    if is_training:
        orig_train_mask = config.TRAIN_MASK_FLIST
        tmp_train_mask = os.path.join(tmp_dir, to_rel(orig_train_mask))
        copy_dir_contents(orig_train_mask, tmp_train_mask)
        config.TRAIN_MASK_FLIST = tmp_train_mask
        
        orig_test_mask = config.TEST_MASK_FLIST
        tmp_test_mask = os.path.join(tmp_dir, to_rel(orig_test_mask))
        copy_dir_contents(orig_test_mask, tmp_test_mask)
        config.TEST_MASK_FLIST = tmp_test_mask
    else:
        orig_mask = config.TEST_MASK_FLIST
        tmp_mask = os.path.join(tmp_dir, to_rel(orig_mask))
        copy_dir_contents(orig_mask, tmp_mask)
        config.TEST_MASK_FLIST = tmp_mask

    # 4. Copy/map image datasets
    if is_training:
        orig_train_images = config.TRAIN_INPAINT_IMAGE_FLIST
        tmp_train_images = os.path.join(tmp_dir, to_rel(orig_train_images))
        
        # Check space before copying full training set
        total_size = 0
        train_files = glob.glob(os.path.join(orig_train_images, '**', '*'), recursive=True)
        train_files = [f for f in train_files if os.path.isfile(f)]
        total_size = sum(os.path.getsize(f) for f in train_files)
        
        usage = shutil.disk_usage(tmp_dir)
        if usage.free < total_size + 2 * 1024 * 1024 * 1024:  # leave 2GB margin
            print(f"WARNING: Not enough space in {tmp_dir} (needed {total_size / (1024**3):.1f} GB, free {usage.free / (1024**3):.1f} GB). Skipping full copy of training images.")
        else:
            copy_dir_contents(orig_train_images, tmp_train_images)
            config.TRAIN_INPAINT_IMAGE_FLIST = tmp_train_images
            
        orig_test_images = config.TEST_INPAINT_IMAGE_FLIST
        tmp_test_images = os.path.join(tmp_dir, to_rel(orig_test_images))
        copy_dir_contents(orig_test_images, tmp_test_images)
        config.TEST_INPAINT_IMAGE_FLIST = tmp_test_images
    else:
        orig_test_images = config.TEST_INPAINT_IMAGE_FLIST
        tmp_test_images = os.path.join(tmp_dir, to_rel(orig_test_images))
        
        limit_names = None
        if selected_images is not None:
            limit_names = set(os.path.basename(f) for f in selected_images)
            
        copy_dir_contents(orig_test_images, tmp_test_images, limit_files=limit_names)
        config.TEST_INPAINT_IMAGE_FLIST = tmp_test_images

    return config, mapped_output


def prepare_tmp_dir_non_zero_rank(config, tmp_dir, is_training=False):
    def to_rel(path):
        import os
        if os.path.isabs(path):
            try:
                return os.path.relpath(path)
            except Exception:
                return path.lstrip('/\\').replace(':', '')
        return path

    if is_training:
        config.PATH = os.path.join(tmp_dir, os.path.basename(config.PATH.rstrip('/\\')))
        config.TRAIN_MASK_FLIST = os.path.join(tmp_dir, to_rel(config.TRAIN_MASK_FLIST))
        config.TEST_MASK_FLIST = os.path.join(tmp_dir, to_rel(config.TEST_MASK_FLIST))
        config.TRAIN_INPAINT_IMAGE_FLIST = os.path.join(tmp_dir, to_rel(config.TRAIN_INPAINT_IMAGE_FLIST))
        config.TEST_INPAINT_IMAGE_FLIST = os.path.join(tmp_dir, to_rel(config.TEST_INPAINT_IMAGE_FLIST))
    else:
        config.PATH = os.path.join(tmp_dir, os.path.basename(config.PATH.rstrip('/\\')))
        config.TEST_MASK_FLIST = os.path.join(tmp_dir, to_rel(config.TEST_MASK_FLIST))
        config.TEST_INPAINT_IMAGE_FLIST = os.path.join(tmp_dir, to_rel(config.TEST_INPAINT_IMAGE_FLIST))

