#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import csv
from collections import defaultdict

def scan_progress(image_source, output_txt_path, output_csv_path):
    print(f"Scanning progress for image source: {image_source}...")
    
    # 1. Load image list
    paths = []
    if os.path.isdir(image_source):
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        for ext in image_extensions:
            paths.extend(glob.glob(os.path.join(image_source, ext)))
            paths.extend(glob.glob(os.path.join(image_source, "**", ext), recursive=True))
        paths = sorted(list(set(paths)))
    elif os.path.isfile(image_source):
        try:
            with open(image_source, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
            base_dir = os.path.dirname(image_source)
            paths = [os.path.join(base_dir, line) if not os.path.isabs(line) else line for line in lines if line.strip()]
        except Exception as e:
            print(f"Error reading flist file {image_source}: {e}")
            sys.exit(1)
    else:
        print(f"Error: Image source {image_source} does not exist.")
        sys.exit(1)
        
    if not paths:
        print(f"No images found for source: {image_source}")
        return

    print(f"Found {len(paths)} total images. Checking segment masks...")
    
    # 2. Count completion per category
    category_counts = defaultdict(lambda: {"total": 0, "completed": 0})
    
    for p in paths:
        img_dir = os.path.dirname(p)
        if os.path.isdir(image_source):
            category = os.path.relpath(img_dir, image_source)
        else:
            category = os.path.basename(img_dir)
            
        base_name, _ = os.path.splitext(os.path.basename(p))
        
        possible_seg_paths = [
            os.path.join(img_dir + "_seg", f"{base_name}.png"),
            os.path.join(os.path.dirname(img_dir), os.path.basename(img_dir) + "_seg", f"{base_name}.png"),
            os.path.join("dataset", "train_seg", f"{base_name}.png"),
            os.path.join("dataset", "test_seg", f"{base_name}.png"),
        ]
        
        seg_exists = False
        for pth in possible_seg_paths:
            if os.path.exists(pth):
                seg_exists = True
                break
                
        category_counts[category]["total"] += 1
        if seg_exists:
            category_counts[category]["completed"] += 1
            
    # 3. Print report and save
    categories = sorted(list(category_counts.keys()))
    
    report_lines = []
    report_lines.append("==============================================================")
    report_lines.append("              PLACES2 SEGMENT MASK GENERATION PROGRESS        ")
    report_lines.append("==============================================================")
    
    csv_rows = []
    
    total_imgs = 0
    total_completed = 0
    
    partially_completed_categories = []
    not_started_categories = []
    completed_categories = []
    
    for cat in categories:
        tot = category_counts[cat]["total"]
        comp = category_counts[cat]["completed"]
        pct = (comp / tot) * 100 if tot > 0 else 0
        
        total_imgs += tot
        total_completed += comp
        
        status = "COMPLETED" if comp == tot else "NOT STARTED" if comp == 0 else "PARTIAL"
        
        if status == "COMPLETED":
            completed_categories.append(cat)
        elif status == "NOT STARTED":
            not_started_categories.append(cat)
        else:
            partially_completed_categories.append((cat, comp, tot, pct))
            
        report_line = f"Category: {cat:<40} | Completed: {comp:5d}/{tot:5d} ({pct:6.2f}%) | Status: {status}"
        report_lines.append(report_line)
        csv_rows.append([cat, comp, tot, f"{pct:.2f}%", status])
        
    overall_pct = (total_completed / total_imgs) * 100 if total_imgs > 0 else 0
    report_lines.append("--------------------------------------------------------------")
    report_lines.append(f"OVERALL PROGRESS: {total_completed}/{total_imgs} images masked ({overall_pct:.2f}%)")
    report_lines.append("--------------------------------------------------------------")
    
    if partially_completed_categories:
        report_lines.append("Currently in-progress (Partial) categories:")
        for cat, comp, tot, pct in partially_completed_categories:
            report_lines.append(f"  - {cat}: {comp}/{tot} ({pct:.2f}%)")
    elif completed_categories and not_started_categories:
        report_lines.append(f"Stopped after fully completing: '{completed_categories[-1]}'")
        report_lines.append(f"Next category to start:        '{not_started_categories[0]}'")
    else:
        report_lines.append("No active partial progress detected (either fully complete or not started).")
        
    report_lines.append("==============================================================")
    
    print("\n".join(report_lines))
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Saved text report to: {output_txt_path}")
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Completed_Masks', 'Total_Images', 'Completion_Percentage', 'Status'])
        writer.writerows(csv_rows)
        writer.writerow(['TOTAL', total_completed, total_imgs, f"{overall_pct:.2f}%", "N/A"])
    print(f"Saved CSV report to: {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and record segment mask generation progress.")
    parser.add_argument("--image-source", default=None, help="Path to image directory or .txt flist file")
    parser.add_argument("--dataset-dir", default="datasets/places365", help="Dataset root directory containing train/val if image-source not specified")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Which split to analyze if using dataset-dir")
    parser.add_argument("--output-txt", default="mask_progress_report.txt", help="Path to save text report")
    parser.add_argument("--output-csv", default="mask_progress_report.csv", help="Path to save CSV report")
    args = parser.parse_args()
    
    image_source = args.image_source
    if not image_source:
        possible_dirs = [
            os.path.join(args.dataset_dir, "places365_standard", args.split),
            os.path.join(args.dataset_dir, args.split),
            args.split
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                image_source = d
                break
        
        if not image_source:
            possible_txts = [
                os.path.join(args.dataset_dir, "places365_standard", f"{args.split}.txt"),
                os.path.join(args.dataset_dir, f"{args.split}.txt"),
                f"{args.split}.txt"
            ]
            for t in possible_txts:
                if os.path.exists(t):
                    image_source = t
                    break
                    
        if not image_source:
            print(f"Error: Could not automatically locate image source for split '{args.split}' in '{args.dataset_dir}'.")
            print("Please specify --image-source directly.")
            sys.exit(1)
            
    scan_progress(image_source, args.output_txt, args.output_csv)

if __name__ == "__main__":
    main()
