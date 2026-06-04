#!/usr/bin/env python3
import os
import glob

def main():
    # Define paths (relative to the repository root on the server)
    train_dir = "datasets/places365/places365_standard/train"
    output_flist = "datasets/places365/places365_standard/train_halved.flist"

    if not os.path.exists(train_dir):
        print(f"Error: Train directory '{train_dir}' does not exist.")
        return

    print(f"Scanning subdirectories in '{train_dir}'...")

    # Collect image files per category
    categories = {}
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    for root, _, _ in os.walk(train_dir):
        img_files = []
        for ext in extensions:
            img_files.extend(glob.glob(os.path.join(root, ext)))
        if img_files:
            img_files.sort()
            categories[root] = img_files

    print(f"Found {len(categories)} categories with images.")

    selected_files = []
    total_found = 0
    for cat, files in categories.items():
        total_found += len(files)
        # Select every second file (approx. 50%)
        selected_files.extend(files[::2])

    print(f"Total images found: {total_found}")
    print(f"Selected images (50%): {len(selected_files)}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_flist), exist_ok=True)

    # Write relative paths (relative to the flist's directory) to avoid duplicate prefixes
    base_dir = os.path.dirname(output_flist)
    with open(output_flist, 'w', encoding='utf-8') as f:
        for filepath in selected_files:
            rel_path = os.path.relpath(filepath, start=base_dir)
            f.write(rel_path + "\n")

    print(f"Successfully wrote halved file list to '{output_flist}'")

if __name__ == "__main__":
    main()
