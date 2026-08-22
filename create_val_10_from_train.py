import os
import glob
import random
import shutil
from collections import defaultdict
import yaml

def main():
    base_train = os.path.abspath("datasets/places365/places365_standard/train")
    if not os.path.exists(base_train):
        base_train = os.path.abspath("datasets/places365/train")
    
    if not os.path.exists(base_train):
        print(f"ERROR: Train directory not found at {base_train}")
        return

    parent_dir = os.path.dirname(base_train)
    val_10_dir = os.path.join(parent_dir, "val_10")
    val_10_seg_dir = os.path.join(parent_dir, "val_10_seg")

    print(f"Scanning training images in {base_train}...")
    all_imgs = glob.glob(os.path.join(base_train, "**", "*.jpg"), recursive=True) + \
               glob.glob(os.path.join(base_train, "**", "*.png"), recursive=True)

    print(f"Found {len(all_imgs)} training images. Filtering for segment masks...")
    
    # Filter for images that have matching segment mask
    valid_samples_by_cat = defaultdict(list)
    for img_path in all_imgs:
        img_dir = os.path.dirname(img_path)
        base_name, _ = os.path.splitext(os.path.basename(img_path))
        possible_seg_paths = [
            os.path.join(img_dir + "_seg", f"{base_name}.png"),
            os.path.join(os.path.dirname(img_dir), os.path.basename(img_dir) + "_seg", f"{base_name}.png"),
            os.path.join(os.path.dirname(img_dir) + "_seg", os.path.basename(img_dir), f"{base_name}.png"),
            os.path.join(os.path.dirname(os.path.dirname(img_dir)) + "_seg", os.path.basename(img_dir), f"{base_name}.png"),
        ]
        seg_path = None
        for pth in possible_seg_paths:
            if os.path.exists(pth):
                seg_path = pth
                break
        if seg_path:
            cat_name = os.path.basename(img_dir)
            valid_samples_by_cat[cat_name].append((img_path, seg_path))

    available_cats = list(valid_samples_by_cat.keys())
    print(f"Found {len(available_cats)} categories with completed segment masks.")
    
    if len(available_cats) == 0:
        print("WARNING: No segment masks found. Will sample 10 images without segment masks.")
        for img_path in all_imgs:
            cat_name = os.path.basename(os.path.dirname(img_path))
            valid_samples_by_cat[cat_name].append((img_path, None))
        available_cats = list(valid_samples_by_cat.keys())

    random.seed(42)  # reproducible selection
    selected_cats = random.sample(available_cats, min(10, len(available_cats)))
    
    # Create output directories
    os.makedirs(val_10_dir, exist_ok=True)
    os.makedirs(val_10_seg_dir, exist_ok=True)

    selected_count = 0
    for cat in selected_cats:
        samples = valid_samples_by_cat[cat]
        img_path, seg_path = random.choice(samples)
        cat_val_dir = os.path.join(val_10_dir, cat)
        cat_val_seg_dir = os.path.join(val_10_seg_dir, cat)
        os.makedirs(cat_val_dir, exist_ok=True)
        os.makedirs(cat_val_seg_dir, exist_ok=True)
        
        dest_img = os.path.join(cat_val_dir, os.path.basename(img_path))
        shutil.copy2(img_path, dest_img)
        
        if seg_path and os.path.exists(seg_path):
            dest_seg = os.path.join(cat_val_seg_dir, os.path.basename(seg_path))
            shutil.copy2(seg_path, dest_seg)
            
        selected_count += 1
        print(f"  [{selected_count}/10] Category '{cat}': {os.path.basename(img_path)}")

    rel_val_10 = os.path.relpath(val_10_dir, os.getcwd()).replace("\\", "/")
    print(f"\nCreated validation set with {selected_count} images at: {rel_val_10}")

    # Update config files
    config_paths = ["PlacesTraining/config.yml", "config.yml"]
    for cfg_p in config_paths:
        if os.path.exists(cfg_p):
            with open(cfg_p, "r") as f:
                cfg = yaml.safe_load(f)
            cfg['TEST_INPAINT_IMAGE_FLIST'] = rel_val_10
            with open(cfg_p, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            print(f"Updated TEST_INPAINT_IMAGE_FLIST to '{rel_val_10}' in {cfg_p}")

if __name__ == "__main__":
    main()
