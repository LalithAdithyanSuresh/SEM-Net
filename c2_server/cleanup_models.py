#!/usr/bin/env python3
import os
import sys
import shutil
import re
import time
import argparse

def check_disk_usage(path):
    try:
        usage = shutil.disk_usage(path)
        percent = (usage.used / usage.total) * 100
        return percent, usage.free
    except Exception as e:
        print(f"Error checking disk usage of {path}: {e}", file=sys.stderr)
        return None, None

def find_model_files(data_dir):
    """
    Scans the data directory recursively for model checkpoint files.
    Matches filename format like: 000002000_InpaintingModel_gen.pth
    Returns a dict mapping iteration (int) -> list of absolute file paths.
    """
    models = {}
    pattern = re.compile(r'^(\d+)_(.*)_(gen|dis)\.pth$')
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            match = pattern.match(file)
            if match:
                iteration = int(match.group(1))
                file_path = os.path.join(root, file)
                if iteration not in models:
                    models[iteration] = []
                models[iteration].append(file_path)
    return models

def run_cleanup(data_dir, threshold, iters_per_epoch, keep_last):
    percent, free = check_disk_usage(data_dir)
    if percent is None:
        return
        
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Disk usage of {data_dir}: {percent:.2f}% (Free: {free / (1024**3):.2f} GB)")
    
    if percent < threshold:
        print("Disk usage is below threshold. No cleanup needed.")
        return
        
    print(f"Disk usage is above threshold ({threshold}%). Starting cleanup...")
    
    models = find_model_files(data_dir)
    if not models:
        print("No model files found to clean.")
        return
        
    sorted_iterations = sorted(models.keys())
    print(f"Found {len(sorted_iterations)} model checkpoint iterations on disk.")
    
    keep_iterations = set()
    
    # 1. Keep the last N iterations (highest iteration numbers)
    last_n = sorted_iterations[-keep_last:]
    keep_iterations.update(last_n)
    
    # 2. Keep one iteration closest to the end of each epoch
    max_iter = sorted_iterations[-1]
    max_epoch = int(max_iter // iters_per_epoch) + 1
    
    for epoch in range(1, max_epoch + 1):
        target_val = epoch * iters_per_epoch
        # Find the iteration closest to the target end of epoch
        closest_iter = min(sorted_iterations, key=lambda x: abs(x - target_val))
        keep_iterations.add(closest_iter)
        
    print(f"Iterations to keep ({len(keep_iterations)}): {sorted(list(keep_iterations))}")
    
    deleted_count = 0
    deleted_size = 0
    
    for iteration in sorted_iterations:
        if iteration not in keep_iterations:
            files_to_delete = models[iteration]
            for file_path in files_to_delete:
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    deleted_size += file_size
                    print(f"  Deleted: {os.path.basename(file_path)} ({file_size / (1024**2):.1f} MB)")
                except Exception as e:
                    print(f"  ERROR deleting {file_path}: {e}", file=sys.stderr)
                    
    print(f"Cleanup finished. Deleted {deleted_count} files, reclaiming {deleted_size / (1024**3):.2f} GB.")
    
    # Re-check usage
    percent_after, _ = check_disk_usage(data_dir)
    if percent_after:
        print(f"New disk usage: {percent_after:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Clean up old model checkpoints to reclaim disk space.")
    parser.add_argument("--data-dir", default=os.path.expanduser("~/data"),
                        help="Data directory containing model checkpoints (default: ~/data)")
    parser.add_argument("--threshold", type=float, default=90.0,
                        help="Disk usage percentage threshold to trigger cleanup (default: 90.0)")
    parser.add_argument("--iters-per-epoch", type=int, default=112000,
                        help="Number of iterations per training epoch (default: 112000)")
    parser.add_argument("--keep-last", type=int, default=5,
                        help="Number of most recent checkpoints to keep (default: 5)")
    parser.add_argument("--daemon", action="store_true",
                        help="Run continuously in the background")
    parser.add_argument("--interval", type=int, default=600,
                        help="Interval in seconds between checks when running as daemon (default: 600)")
                        
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory '{args.data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    if args.daemon:
        print(f"Starting model cleanup daemon. Monitoring {args.data_dir} every {args.interval}s...")
        try:
            while True:
                run_cleanup(args.data_dir, args.threshold, args.iters_per_epoch, args.keep_last)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nDaemon stopped by user.")
    else:
        run_cleanup(args.data_dir, args.threshold, args.iters_per_epoch, args.keep_last)

if __name__ == "__main__":
    main()
