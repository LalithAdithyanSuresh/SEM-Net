import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask
import cv2
from skimage.feature import canny

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training

        self.data = self.load_flist(flist, is_mask=False)
        self.mask_data = self.load_flist(mask_flist, is_mask=True)


        self.input_size = config.INPUT_SIZE
        self.mask = config.MASK
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size


        # load image
        img = imread(self.data[index])

       
        
        if len(img.shape) < 3:
            img = gray2rgb(img)

        
        
        if size != 0:
            img = self.resize(img, size, size, centerCrop=True)

        ### for large image inpainting.
        
        #imgh, imgw = img.shape[0:2]
        #if imgh % 16 != 0:
        #    imgh -= imgh % 16
        
        #if imgw % 16 != 0:
        #    imgw -= imgw % 16
        #img = self.resize(img, 2560, 1920, centerCrop=False)
        

        # load mask
        mask = self.load_mask(img, index)

        # load segment map (unique instance IDs)
        seg_map = self.load_seg_map(self.data[index], size)

        return self.to_tensor(img), self.to_tensor(mask), self.to_tensor(seg_map)

    def load_seg_map(self, img_path, size):
        img_dir = os.path.dirname(img_path)
        base_name, _ = os.path.splitext(os.path.basename(img_path))
        
        possible_seg_paths = [
            os.path.join(img_dir + "_seg", f"{base_name}.png"),
            os.path.join(os.path.dirname(img_dir), os.path.basename(img_dir) + "_seg", f"{base_name}.png"),
            os.path.join("dataset", "train_seg", f"{base_name}.png"),
            os.path.join("dataset", "test_seg", f"{base_name}.png"),
            # Support train_seg/abbey/00000001.png
            os.path.join(os.path.dirname(img_dir) + "_seg", os.path.basename(img_dir), f"{base_name}.png"),
            # Support train_seg/00000001.png
            os.path.join(os.path.dirname(img_dir) + "_seg", f"{base_name}.png"),
            # Support /path/to/train_seg/00000001.png when nested in category abbey
            os.path.join(os.path.dirname(os.path.dirname(img_dir)) + "_seg", f"{base_name}.png"),
            # Support /path/to/train_seg/abbey/00000001.png when nested in category abbey
            os.path.join(os.path.dirname(os.path.dirname(img_dir)) + "_seg", os.path.basename(img_dir), f"{base_name}.png"),
        ]
        
        seg_path = None
        for pth in possible_seg_paths:
            if os.path.exists(pth):
                seg_path = pth
                break
                
        if seg_path and os.path.exists(seg_path):
            try:
                seg_img = imread(seg_path)
                if len(seg_img.shape) == 3:
                    seg_img = seg_img[:, :, 0]
                if size != 0:
                    seg_img = self.resize(seg_img, size, size, centerCrop=True)
                return seg_img
            except Exception:
                pass
                
        h, w = (size, size) if size != 0 else (256, 256)
        return np.zeros((h, w), dtype=np.uint8)


    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask


        if mask_type == 5:
            mask_type = 0 if np.random.uniform(0,1) >= 0.5 else 4

        # no mask
        if mask_type == 0:
            return np.zeros((self.config.INPUT_SIZE,self.config.INPUT_SIZE))

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # center mask
        if mask_type == 2:
            return create_mask(imgw, imgh, imgw//2, imgh//2, x = imgw//4, y = imgh//4)

        # external
        if mask_type == 3:
            if len(self.mask_data) == 0:
                mask = 1 - generate_stroke_mask([imgh, imgw])
                mask = (mask > 0).astype(np.uint8) * 255
                mask = self.resize(mask, imgh, imgw, centerCrop=False)
                return mask
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            #mask = (mask > 100).astype(np.uint8) * 255
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            if len(self.mask_data) == 0:
                mask = 1 - generate_stroke_mask([imgh, imgw])
                mask = (mask > 0).astype(np.uint8) * 255
                mask = self.resize(mask, imgh, imgw, centerCrop=False)
                return mask
            mask = imread(self.mask_data[index%len(self.mask_data)])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            #mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        # random mask
        if mask_type ==7:
                mask = 1 - generate_stroke_mask([imgh, imgw])
                mask = (mask > 0).astype(np.uint8) * 255
                mask = self.resize(mask, imgh, imgw, centerCrop=False)
                return mask

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask).astype(float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        # img = scipy.misc.imresize(img, [height, width])
        img = np.array(Image.fromarray(img).resize((height, width)))
        return img

    def load_flist(self, flist, is_mask=False):
        if isinstance(flist, list):
            paths = flist
        elif isinstance(flist, str):
            if os.path.isdir(flist):
                paths = list(glob.glob(os.path.join(flist, '**', '*.jpg'), recursive=True)) + \
                        list(glob.glob(os.path.join(flist, '**', '*.png'), recursive=True))
                paths.sort()
            elif os.path.isfile(flist):
                try:
                    data = np.genfromtxt(flist, dtype=str, encoding='utf-8')
                    if data.ndim == 0:
                        data = np.array([data])
                    
                    base_dir = os.path.dirname(flist)
                    paths = [os.path.join(base_dir, line) if not os.path.isabs(line) else line for line in data]
                except Exception as e:
                    print(e)
                    paths = [flist]
            else:
                paths = []
        else:
            paths = []

        if getattr(self.config, 'FILTER_BY_SEG_MASK', False) and not is_mask and len(paths) > 0:
            filtered_paths = []
            for p in paths:
                img_dir = os.path.dirname(p)
                base_name, _ = os.path.splitext(os.path.basename(p))
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
                seg_exists = False
                for pth in possible_seg_paths:
                    if os.path.exists(pth):
                        seg_exists = True
                        break
                if seg_exists:
                    filtered_paths.append(p)
            print(f"[DATASET] FILTER_BY_SEG_MASK is enabled: filtered images from {len(paths)} down to {len(filtered_paths)} (kept only images with generated segment masks).")
            paths = filtered_paths

        if self.training and len(paths) > 0:
            from collections import defaultdict
            categories = defaultdict(list)
            for p in paths:
                categories[os.path.dirname(p)].append(p)

            sorted_cat_dirs = sorted(categories.keys())
            max_cats = getattr(self.config, 'MAX_CATEGORIES', None)
            if max_cats is not None and max_cats > 0:
                sorted_cat_dirs = sorted_cat_dirs[:max_cats]

            selected_paths = []
            for cat_dir in sorted_cat_dirs:
                cat_files = sorted(categories[cat_dir])
                selected_paths.extend(cat_files)

            print(f"[DATASET] Filtered to first {len(sorted_cat_dirs)} categories (full dataset): total {len(selected_paths)} training images.")
            return selected_paths
        
        return paths

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                num_workers=8,
                pin_memory=True,
                shuffle=True
            )

            for item in sample_loader:
                yield item


def generate_stroke_mask(im_size, max_parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    return mask

def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask

def image_transforms(load_size):

    return transforms.Compose([

        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
