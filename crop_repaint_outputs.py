import os
from PIL import Image

src_dir = r"d:\_CAPSTONE\ImageInPainting\SEM-Net\imagesFromRePaint"
dst_dir = r"d:\_CAPSTONE\ImageInPainting\SEM-Net\imagesFromRePaint_cropped"

os.makedirs(dst_dir, exist_ok=True)

files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
for f in files:
    img_path = os.path.join(src_dir, f)
    try:
        img = Image.open(img_path)
        w, h = img.size
        
        # 5 images attached horizontally. The 5th is the rightmost.
        single_w = w // 5
        # Crop the 5th image
        # box is (left, upper, right, lower)
        box = (4 * single_w, 0, w, h)
        cropped = img.crop(box)
        
        out_path = os.path.join(dst_dir, f)
        cropped.save(out_path)
        print(f"Cropped {f}: original size {w}x{h}, cropped to 5th image size {cropped.size}")
    except Exception as e:
        print(f"Error processing {f}: {e}")
