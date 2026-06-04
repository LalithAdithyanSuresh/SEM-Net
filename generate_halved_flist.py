#!/usr/bin/env python3
import os
import glob

def main():
    # Define paths (relative or absolute as needed on the server)
    train_dir = "datasets/places365/places365_standard/train"
    output_flist = "datasets/places365/places365_standard/train_halved.flist"

    if not os.path.exists(train_dir):
        print(f"Error: Train directory '{train_dir}' does not exist.")
        return

    print(f"Scanning subdirectories in '{train_dir}'...")
    
    # We walk the directory and find all subdirectories that contain images
    categories = {}
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    
    # Find all subdirs
    for root, dirs, files in os.walk(train_dir):
        # Filter files by extensions
        img_files = []
        for ext in extensions:
            img_files.extend(glob.glob(os.path.join(root, ext)))
        
        if img_files:
            # Sort files deterministically
            img_files.sort()
            categories[root] = img_files

    print(f"Found {len(categories)} categories with images.")
    
    selected_files = []
    total_found = 0
    
    for cat, files in categories.items():
        total_found += len(files)
        # Select 50% of files (every 2nd file)
        half_files = files[::2]
        selected_files.extend(half_files)
        
    print(f"Total images found: {total_found}")
    print(f"Selected images (50%): {len(selected_files)}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_flist), exist_ok=True)
    
    # Write to flist file
    with open(output_flist, 'w', encoding='utf-8') as f:
        for filepath in selected_files:
            # We can write relative paths
            f.write(filepath + "\n")
            
    print(f"Successfully wrote halved file list to '{output_flist}'")

if __name__ == "__main__":
    main()
