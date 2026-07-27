import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image

def draw_header(img, title):
    """Adds a stylish header bar at the top of an image panel."""
    h, w, c = img.shape
    bar_h = 30
    header_img = np.zeros((h + bar_h, w, c), dtype=np.uint8)
    header_img[bar_h:] = img
    
    # Draw dark header background
    cv2.rectangle(header_img, (0, 0), (w, bar_h), (30, 30, 30), -1)
    
    # Draw centered text
    text_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    text_x = max(2, (w - text_size[0]) // 2)
    text_y = (bar_h + text_size[1]) // 2
    cv2.putText(header_img, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    
    return header_img

def process_masks_to_colored_map(raw_masks, h, w, filter_rectangles=True):
    """
    Converts a array of binary masks [N, H, W] into a colored segment instance map.
    Embeds unique instance ID (1..255) in Channel 0 and distinct visual RGB in Channels 1 & 2.
    """
    visual_map = np.zeros((h, w, 3), dtype=np.uint8)
    if raw_masks is None or len(raw_masks) == 0:
        return visual_map

    valid_masks = []
    for m in raw_masks:
        m_bin = m > 0.5
        mask_area = np.sum(m_bin)
        if mask_area < 25:
            continue
        if filter_rectangles:
            y_idx, x_idx = np.where(m_bin)
            if len(y_idx) == 0:
                continue
            bbox_area = (y_idx.max() - y_idx.min() + 1) * (x_idx.max() - x_idx.min() + 1)
            rectangularity = mask_area / float(bbox_area)
            if rectangularity > 0.88:
                continue
        valid_masks.append(m_bin)

    if valid_masks:
        np.random.seed(42)
        num_masks = len(valid_masks)
        colors = np.random.randint(40, 255, size=(max(num_masks, 200), 3), dtype=np.uint8)
        
        areas = [np.sum(m) for m in valid_masks]
        sorted_indices = np.argsort(areas)[::-1]
        
        for rank, idx in enumerate(sorted_indices):
            mask_i = valid_masks[idx]
            unique_id = int((rank + 1) * 255 / max(num_masks, 1))
            visual_map[mask_i, 0] = unique_id
            visual_map[mask_i, 1:] = colors[idx, 1:]
            
    return visual_map

def generate_fastsam_map(model, img_path, device="cuda"):
    res = model(img_path, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
    if res and len(res) > 0 and res[0].masks is not None and len(res[0].masks.data) > 0:
        raw_masks = res[0].masks.data.cpu().numpy()
        h, w = raw_masks.shape[1], raw_masks.shape[2]
        return process_masks_to_colored_map(raw_masks, h, w, filter_rectangles=True)
    img = cv2.imread(img_path)
    h, w = img.shape[:2] if img is not None else (256, 256)
    return np.zeros((h, w, 3), dtype=np.uint8)

def generate_full_sam_map(sam_model, img_path, device="cuda"):
    """Generates segment map using original/full SAM (or MobileSAM)."""
    try:
        res = sam_model(img_path, device=device, verbose=False)
        if res and len(res) > 0 and res[0].masks is not None and len(res[0].masks.data) > 0:
            raw_masks = res[0].masks.data.cpu().numpy()
            h, w = raw_masks.shape[1], raw_masks.shape[2]
            return process_masks_to_colored_map(raw_masks, h, w, filter_rectangles=False)
    except Exception as e:
        print(f"   [SAM Warning] {e}")
        
    img = cv2.imread(img_path)
    h, w = img.shape[:2] if img is not None else (256, 256)
    return np.zeros((h, w, 3), dtype=np.uint8)

def main():
    print("=" * 65)
    print("  FastSAM vs Full SAM 3-Column Segment Comparison Visualizer  ")
    print("=" * 65)
    
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

    print(f"-> Found {len(image_paths)} images for testing.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Load FastSAM & Full SAM models
    from ultralytics import FastSAM, SAM
    print(f"-> Loading FastSAM model on {device}...")
    fastsam_model = FastSAM("FastSAM-s.pt")
    
    print(f"-> Loading Full SAM model (sam_b.pt) on {device}...")
    try:
        full_sam_model = SAM("sam_b.pt")
    except Exception:
        print("-> Fallback: Loading MobileSAM (mobile_sam.pt)...")
        full_sam_model = SAM("mobile_sam.pt")
    
    target_size = (256, 256)
    panels = []
    
    print("-> Processing 10 images across FastSAM and Full SAM...")
    for idx, img_path in enumerate(image_paths, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load GT image
        gt_bgr = cv2.imread(img_path)
        if gt_bgr is None:
            continue
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_resized = cv2.resize(gt_rgb, target_size, interpolation=cv2.INTER_AREA)
        
        # Generate FastSAM Map
        fast_map_rgb = generate_fastsam_map(fastsam_model, img_path, device=device)
        fast_resized = cv2.resize(fast_map_rgb, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Generate Full SAM Map
        sam_map_rgb = generate_full_sam_map(full_sam_model, img_path, device=device)
        sam_resized = cv2.resize(sam_map_rgb, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Format panels with headers
        gt_panel   = draw_header(gt_resized,   f"GT #{idx}: {base_name}")
        fast_panel = draw_header(fast_resized, f"FastSAM #{idx}")
        sam_panel  = draw_header(sam_resized,  f"Full SAM (sam_b) #{idx}")
        
        # Combine side-by-side (Row: [GT | FastSAM | Full SAM])
        row_img = np.hstack([gt_panel, fast_panel, sam_panel])
        panels.append(row_img)
        print(f"   [✓] Image {idx}/{len(image_paths)} processed: {base_name}")

    # 3. Create 3-Column x 10-Row Collage
    if not panels:
        print("[Error] No panels produced.")
        return
        
    collage_img = np.vstack(panels)
    
    # Add top banner
    banner_h = 45
    full_w = collage_img.shape[1]
    banner = np.zeros((banner_h, full_w, 3), dtype=np.uint8)
    cv2.rectangle(banner, (0, 0), (full_w, banner_h), (20, 20, 20), -1)
    
    title_text = "Segment Map Comparison (Col 1: GT | Col 2: FastSAM | Col 3: Full SAM)"
    cv2.putText(banner, title_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2, cv2.LINE_AA)
    
    final_collage = np.vstack([banner, collage_img])
    
    output_collage_path = "segment_test_collage.png"
    cv2.imwrite(output_collage_path, cv2.cvtColor(final_collage, cv2.COLOR_RGB2BGR))
    
    print("=" * 65)
    print(f"SUCCESS: 3-Column collage saved to: {os.path.abspath(output_collage_path)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
