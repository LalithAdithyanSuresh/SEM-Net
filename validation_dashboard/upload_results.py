import os
import sys
import shutil
import requests
import argparse
import time
import math
import sys

def upload_chunked(zip_path, host):
    CHUNK_SIZE = 50 * 1024 * 1024  # 50MB
    file_size = os.path.getsize(zip_path)
    total_chunks = math.ceil(file_size / CHUNK_SIZE)
    filename = os.path.basename(zip_path)
    upload_url = f"{host.rstrip('/')}/api/upload_chunk"
    
    print(f"Uploading {zip_path} ({file_size / 1e6:.1f} MB) in {total_chunks} chunks...\n")
    
    start_time = time.time()
    
    with open(zip_path, 'rb') as f:
        for i in range(total_chunks):
            chunk_data = f.read(CHUNK_SIZE)
            retries = 3
            
            for attempt in range(retries):
                try:
                    progress = (i + 1) / total_chunks * 100
                    elapsed = time.time() - start_time
                    
                    if i > 0:
                        speed = (i * CHUNK_SIZE) / elapsed
                        eta = ((total_chunks - i) * CHUNK_SIZE) / speed if speed > 0 else 0
                        eta_mins, eta_secs = divmod(int(eta), 60)
                        eta_str = f"{eta_mins}m {eta_secs}s"
                    else:
                        eta_str = "Calculating..."
                        
                    status_line = f"\rUploading chunk {i+1}/{total_chunks} ({progress:.1f}%) - ETA: {eta_str}"
                    if attempt > 0:
                        status_line += f" [Attempt {attempt+1}]"
                    sys.stdout.write(status_line.ljust(80))
                    sys.stdout.flush()
                    
                    files = {'file': (filename, chunk_data, 'application/octet-stream')}
                    data = {
                        'filename': filename,
                        'chunk_index': i,
                        'total_chunks': total_chunks
                    }
                    
                    # 60s timeout for a 50MB chunk
                    response = requests.post(upload_url, files=files, data=data, timeout=60)
                    
                    if response.status_code == 200:
                        break
                    else:
                        print(f"\n  -> Failed with status {response.status_code}: {response.text}")
                        if attempt == retries - 1:
                            print("\nMax retries reached. Upload aborted.")
                            return
                        time.sleep(2)
                except requests.exceptions.RequestException as e:
                    print(f"\n  -> Network error: {e}")
                    if attempt == retries - 1:
                        print("\nMax retries reached. Upload aborted.")
                        return
                    time.sleep(2)
                    
    print("\n\nUpload and server-side extraction successful!")

def upload_path(path, host, target_name=None):
    if not host.startswith(('http://', 'https://')):
        host = 'http://' + host # Default to http for local/custom domains

    cleanup = False
    if os.path.isdir(path):
        folder_name = target_name or os.path.basename(os.path.normpath(path))
        zip_path = f"{folder_name}.zip"
        print(f"Zipping {path} to {zip_path}...")
        shutil.make_archive(folder_name, 'zip', path)
        cleanup = True
    elif os.path.isfile(path) and path.endswith('.zip'):
        zip_path = path
        print(f"Using existing zip file: {zip_path}")
    else:
        print(f"Error: {path} must be a valid directory or a .zip file.")
        return
        
    try:
        upload_chunked(zip_path, host)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if cleanup and os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up temporary zip file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload validation results to the dashboard server.")
    parser.add_argument("path", nargs='?', help="Path to the folder or .zip file containing results")
    parser.add_argument("--host", default="http://validate.lalithadithyan.dev", help="URL of the dashboard server")
    parser.add_argument("--masks", help="Path to the original mask directory to upload")
    
    args = parser.parse_args()
    
    if args.masks:
        print(f"Targeting masks folder: {args.masks}")
        upload_path(args.masks, args.host, target_name="masks")
    
    if args.path:
        upload_path(args.path, args.host)
    elif not args.masks:
        parser.print_help()
