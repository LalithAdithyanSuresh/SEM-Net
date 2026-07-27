import os
import requests
import time
import math
import sys

def create_dummy_file(file_path, size_in_mb):
    print(f"Creating dummy file of size {size_in_mb} MB...")
    block = b"\0" * (1024 * 1024)  # 1MB block
    with open(file_path, "wb") as f:
        for _ in range(size_in_mb):
            f.write(block)

def run_upload_test(file_path, chunk_size, server_url, session_id, timeout_sec):
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    total_chunks = math.ceil(file_size / chunk_size)
    
    print(f"\n[TEST] Uploading {filename} ({file_size / (1024*1024):.1f} MB) in {total_chunks} chunks (size: {chunk_size / (1024*1024):.1f} MB)...")
    
    start_time = time.time()
    try:
        with open(file_path, 'rb') as f:
            for i in range(total_chunks):
                chunk_data = f.read(chunk_size)
                files = {'file': (f"{filename}.part{i}", chunk_data, 'application/octet-stream')}
                data = {
                    'session': session_id,
                    'filename': filename,
                    'chunk_index': i,
                    'total_chunks': total_chunks
                }
                
                chunk_start = time.time()
                res = requests.post(f"{server_url}/api/upload_chunk", files=files, data=data, timeout=timeout_sec)
                chunk_elapsed = time.time() - chunk_start
                
                if res.status_code == 200:
                    speed = (len(chunk_data) / (1024*1024)) / chunk_elapsed
                    print(f"   -> Chunk {i+1}/{total_chunks} uploaded successfully in {chunk_elapsed:.2f}s ({speed:.2f} MB/s)")
                else:
                    print(f"   -> Chunk {i+1}/{total_chunks} failed. Status: {res.status_code}, Response: {res.text[:200]}")
                    return False
        
        total_elapsed = time.time() - start_time
        overall_speed = (file_size / (1024*1024)) / total_elapsed
        print(f"[SUCCESS] Uploaded {filename} in {total_elapsed:.2f}s (Average Speed: {overall_speed:.2f} MB/s)")
        return True
    except Exception as e:
        print(f"[FAILED] Error during upload: {e}")
        return False

def main():
    print("Network lab upload sweep is disabled.")
    return
