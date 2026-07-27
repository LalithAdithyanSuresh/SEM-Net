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
    def __init__(self, config, flist, segment_flist=None, mask_flist=None, augment=True, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training

        self.data = self.load_flist(flist)
        self.segment_flist = segment_flist
        self.segment_data = self.load_flist(segment_flist) if segment_flist else None
        self.mask_data = self.load_flist(mask_flist) if mask_flist else None

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

        # load mask
        mask = self.load_mask(img, index)

        # load SAM segment map if present
        seg_img = self.load_segment(img, index)
        if seg_img is not None:
            return self.to_tensor(img), self.to_tensor(mask), self.to_tensor(seg_img)

        return self.to_tensor(img), self.to_tensor(mask)

    def load_segment(self, img, index):
        if not self.segment_data or len(self.segment_data) == 0:
            return None
        try:
            if index < len(self.segment_data):
                seg_path = self.segment_data[index]
            else:
                img_name = os.path.basename(self.data[index])
                seg_path = os.path.join(self.segment_flist, img_name) if isinstance(self.segment_flist, str) else self.data[index]

            if os.path.exists(seg_path):
                seg = imread(seg_path)
                if len(seg.shape) < 3:
                    seg = gray2rgb(seg)
                if self.input_size != 0:
                    seg = self.resize(seg, self.input_size, self.input_size, centerCrop=True)
                return seg
        except Exception:
            pass
        return None

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
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index%len(self.mask_data)])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        # random mask
        if mask_type ==7:
            mask = 1 - generate_stroke_mask([imgh, imgw])
            mask = (mask > 0).astype(np.uint8) * 255
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = np.array(Image.fromarray(img).resize((height, width)))
        return img

    def load_flist(self, flist):
        if not flist:
            return []
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
    num_parts = np.random.randint(1, max_parts + 1)
    for i in range(num_parts):
        num_vertex = np.random.randint(1, maxVertex + 1)
        angle = np.random.randint(maxAngle)
        length = np.random.randint(maxLength)
        brush_width = np.random.randint(10, maxBrushWidth + 1)
        start_x = np.random.randint(im_size[1])
        start_y = np.random.randint(im_size[0])

        for j in range(num_vertex):
            angle = angle + np.random.randint(-20, 21)
            end_x = int(start_x + length * np.cos(np.radians(angle)))
            end_y = int(start_y + length * np.sin(np.radians(angle)))

            cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_width)
            cv2.circle(mask, (start_x, start_y), brush_width // 2, 1.0)
            cv2.circle(mask, (end_x, end_y), brush_width // 2, 1.0)

            start_x, start_y = end_x, end_y

    return mask
