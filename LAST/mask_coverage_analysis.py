import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

def analyze_mask_coverage(mask_dir, output_csv):
    print(f"Scanning masks in: {mask_dir}")
    
    # Supported image extensions
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(exts)]
    mask_files.sort()
    
    results = []
    
    print(f"Calculating coverage for {len(mask_files)} masks...")
    for filename in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, filename)
        
        try:
            # Load as grayscale
            mask_img = Image.open(mask_path).convert('L')
            mask_np = np.array(mask_img)
            
            # Coverage percentage: (white pixels / total pixels) * 100
            # Assuming masks are 0 (black) and 255 (white)
            coverage = (np.mean(mask_np) / 255.0) * 100
            
            results.append([filename, f"{coverage:.4f}"])
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Write to CSV
    print(f"Saving results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mask_Name', 'Coverage_Percentage'])
        writer.writerows(results)
    
    print("Analysis complete.")

if __name__ == "__main__":
    # Path to your mask folder
    MASK_FOLDER = "/mnt/datadrive/inpaint/iregularmask/test_mask/mask/testing_mask_dataset"
    OUTPUT_FILE = "mask_coverage_report.csv"
    
    if os.path.exists(MASK_FOLDER):
        analyze_mask_coverage(MASK_FOLDER, OUTPUT_FILE)
    else:
        print(f"Error: Mask folder '{MASK_FOLDER}' not found. Please update the MASK_FOLDER path in the script.")
