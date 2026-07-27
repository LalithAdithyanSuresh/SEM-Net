import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_header(img, title, font_size=16):
    """Adds a stylish header bar at the top of an image panel."""
    h, w, c = img.shape
    bar_h = 30
    header_img = np.zeros((h + bar_h, w, c), dtype=np.uint8)
    header_img[bar_h:] = img
    
    # Draw dark header background
    cv2.rectangle(header_img, (0, 0), (w, bar_h), (30, 30, 30), -1)
    
    # Draw centered text
    text_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_x = (w - text_size[0]) // 2
    text_y = (bar_h + text_size[1]) // 2
    cv2.putText(header_img, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    
    return header_img

def generate_segment_map_fastsam(model, img_path, device="cuda"):
    """
    Generates high-quality segment map with FastSAM instance segmentation.
    Returns:
       visual_map: 3-channel RGB image with distinct instance colors & organic shapes
    """
    res = model(img_path, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
    
    if res and len(res) > 0 and res[0].masks is not None and len(res[0].masks.data) > 0:
        raw_masks = res[0].masks.data.cpu().numpy() # [N, H, W]
        h, w = raw_masks.shape[1], raw_masks.shape[2]
        visual_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Filter out rectangular/square bounding box masks
        organic_masks = []
        for m in raw_masks:
            m_bin = m > 0.5
            mask_area = np.sum(m_bin)
            if mask_area < 25:
                continue
            y_idx, x_idx = np.where(m_bin)
            if len(y_idx) == 0:
                continue
            bbox_area = (y_idx.max() - y_idx.min() + 1) * (x_idx.max() - x_idx.min() + 1)
            rectangularity = mask_area / float(bbox_area)
            if rectangularity <= 0.88:
                organic_masks.append(m_bin)
                
        if organic_masks:
            np.random.seed(42)
            num_masks = len(organic_masks)
            colors = np.random.randint(40, 255, size=(max(num_masks, 200), 3), dtype=np.uint8)
            
            areas = [np.sum(m) for m in organic_masks]
            sorted_indices = np.argsort(areas)[::-1]
            
            for rank, idx in enumerate(sorted_indices):
                mask_i = organic_masks[idx]
                unique_id = int((rank + 1) * 255 / max(num_masks, 1))
                visual_map[mask_i, 0] = unique_id
                visual_map[mask_i, 1:] = colors[idx, 1:]
                
        return visual_map
    else:
        img = cv2.imread(img_path)
        h, w = img.shape[:2] if img is not None else (256, 256)
        return np.zeros((h, w, 3), dtype=np.uint8)

def main():
    print("=" * 60)
    print("      FastSAM 10-Image Segment Map Visualizer & Tester      ")
    print("=" * 60)
    
    # 1. Search for test images
    test_dirs = ["dataset/test", "dataset/train", "test", "train"]
    image_paths = []
    for t_dir in test_dirs:
        if os.path.exists(t_dir):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                image_paths.extend(glob.glob(os.path.join(t_dir, ext)))
                image_paths.extend(glob.glob(os.path.join(t_dir, "**", ext), recursive=True))
        if len(image_paths) >= 10:
            break
            
    image_paths = sorted(list(set(image_paths)))[:10]
    
    if not image_paths:
        print("[Error] No test images found in dataset directories!")
        return

    print(f"-> Found {len(image_paths)} images for fast testing.")
    for idx, p in enumerate(image_paths, 1):
        print(f"   [{idx:02d}] {p}")
        
    # 2. Load FastSAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Loading FastSAM model on {device}...")
    from ultralytics import FastSAM
    model = FastSAM("FastSAM-s.pt")
    
    output_dir = "dataset/test_seg_sample"
    os.makedirs(output_dir, exist_ok=True)
    
    panels = []
    target_size = (256, 256)
    
    print("-> Processing images and generating segment maps...")
    for idx, img_path in enumerate(image_paths, 1):
        # Load GT image
        gt_bgr = cv2.imread(img_path)
        if gt_bgr is None:
            continue
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_resized = cv2.resize(gt_rgb, target_size, interpolation=cv2.INTER_AREA)
        
        # Generate Segment Map
        seg_map_rgb = generate_segment_map_fastsam(model, img_path, device=device)
        seg_resized = cv2.resize(seg_map_rgb, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Save individual sample mask
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        sample_save_path = os.path.join(output_dir, f"{base_name}.png")
        cv2.imwrite(sample_save_path, cv2.cvtColor(seg_resized, cv2.COLOR_RGB2BGR))
        
        # Format panel pair with headers
        gt_panel = draw_header(gt_resized, f"GT #{idx}: {base_name}")
        seg_panel = draw_header(seg_resized, f"FastSAM Segment Map #{idx}")
        
        # Combine side-by-side (Row: [GT | Segment Map])
        row_img = np.hstack([gt_panel, seg_panel])
        panels.append(row_img)
        print(f"   [✓] Image {idx}/{len(image_paths)} processed -> Saved sample: {sample_save_path}")

    # 3. Create 2-Column x 10-Row Collage
    if not panels:
        print("[Error] No panels produced.")
        return
        
    collage_img = np.vstack(panels)
    
    # Add top banner
    banner_h = 45
    full_w = collage_img.shape[1]
    banner = np.zeros((banner_h, full_w, 3), dtype=np.uint8)
    cv2.rectangle(banner, (0, 0), (full_w, banner_h), (20, 20, 20), -1)
    
    title_text = "FastSAM Segment Map Verification (Col 1: GT | Col 2: Segment Map)"
    cv2.putText(banner, title_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
    
    final_collage = np.vstack([banner, collage_img])
    
    output_collage_path = "segment_test_collage.png"
    cv2.imwrite(output_collage_path, cv2.cvtColor(final_collage, cv2.COLOR_RGB2BGR))
    
    print("=" * 60)
    print(f"SUCCESS: Collage created and saved to: {os.path.abspath(output_collage_path)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
