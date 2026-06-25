import os
import csv

def generate_flists():
    categories = ['SMALL', 'MEDIUM', 'LARGE']
    csv_dir = r"d:\_CAPSTONE\ImageInPainting\SEM-Net\CCC\CMT_Validate_Places2\results"
    output_dir = r"d:\_CAPSTONE\ImageInPainting\SEM-Net\datasets\places365"
    os.makedirs(output_dir, exist_ok=True)
    
    for cat in categories:
        csv_path = os.path.join(csv_dir, f"metrics_{cat}.csv")
        img_flist_path = os.path.join(output_dir, f"val_images_{cat}.flist")
        mask_flist_path = os.path.join(output_dir, f"val_masks_{cat}.flist")
        
        print(f"Processing category {cat} from {csv_path}...")
        
        images = []
        masks = []
        
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Image,MaskID,PSNR,SSIM,L1,LPIPS
            
            for row in reader:
                if not row or not row[0].strip() or row[0] == 'AVERAGE':
                    continue
                
                img_name = row[0]
                mask_id = row[1]
                
                # Image relative path to datasets/places365/
                img_rel = f"test_256/{img_name}"
                # Mask relative path to datasets/places365/
                mask_rel = f"../testing_mask_dataset/{mask_id}.png"
                
                # Verify that files actually exist locally
                full_img_path = os.path.join(output_dir, img_rel)
                full_mask_path = os.path.join(output_dir, mask_rel)
                
                if not os.path.exists(full_img_path):
                    print(f"Warning: Image {full_img_path} does not exist!")
                if not os.path.exists(full_mask_path):
                    print(f"Warning: Mask {full_mask_path} does not exist!")
                
                images.append(img_rel)
                masks.append(mask_rel)
                
        # Write image flist
        with open(img_flist_path, 'w', encoding='utf-8') as out_f:
            for item in images:
                out_f.write(item + '\n')
                
        # Write mask flist
        with open(mask_flist_path, 'w', encoding='utf-8') as out_f:
            for item in masks:
                out_f.write(item + '\n')
                
        print(f"Wrote {len(images)} lines to {img_flist_path}")
        print(f"Wrote {len(masks)} lines to {mask_flist_path}")

if __name__ == "__main__":
    generate_flists()
