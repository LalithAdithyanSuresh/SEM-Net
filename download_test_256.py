#!/usr/bin/env python3
"""
Download test_256.zip for Places365
====================================
Downloads test_256.zip from files.lalithadithyan.dev and extracts it to
datasets/places365/test_256.
"""

import os
import sys
import urllib.request
import time
import zipfile

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB
        
        last_update_time = time.time()
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            
            current_time = time.time()
            if current_time - last_update_time > 5.0 or downloaded == total_size:
                last_update_time = current_time
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    msg = f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)"
                else:
                    msg = f"Downloaded {downloaded / (1024*1024):.1f} MB"
                print(msg)
                sys.stdout.flush()

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed successfully.")

def main():
    url = "https://files.lalithadithyan.dev/download/test_256.zip"
    dest_zip = os.path.join("datasets", "places365", "test_256.zip")
    extract_to = os.path.join("datasets", "places365")
    
    try:
        download_file(url, dest_zip)
        extract_zip(dest_zip, extract_to)
        
        # Clean up zip file
        if os.path.exists(dest_zip):
            print("Cleaning up temporary zip file...")
            os.remove(dest_zip)
            
        print("\n[✓] Done! test_256 directory is now set up under datasets/places365/test_256.")
    except Exception as e:
        print(f"\n[ERROR] Failed to download or extract test_256.zip: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
