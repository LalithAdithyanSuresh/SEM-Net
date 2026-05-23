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
    server_url = "https://files.lalithadithyan.dev"
    session_id = "lab_sweep_test"
    
    # Sweep sizes (in Megabytes)
    test_sizes_mb = [5, 10, 50, 100, 250, 500]
    
    results = {}
    
    print("="*60)
    print("          LAB SERVER UPLOAD SWEEP BENCHMARK")
    print(f" Target Server: {server_url}")
    print("="*60)
    
    for mb in test_sizes_mb:
        chunk_size_bytes = mb * 1024 * 1024
        
        # Generous timeout for high-speed link
        timeout_sec = 300 
        
        # File is 2 chunks in size (so it verifies chunking and assembly)
        test_file = f"test_{mb}mb.bin"
        file_size_mb = mb * 2
        
        print(f"\n" + "-"*50)
        print(f"Testing Chunk Size: {mb} MB (File size: {file_size_mb} MB)")
        print(f"-"*50)
        
        create_dummy_file(test_file, file_size_mb)
        
        success = run_upload_test(test_file, chunk_size_bytes, server_url, session_id, timeout_sec)
        
        # Clean up file immediately
        if os.path.exists(test_file):
            os.remove(test_file)
            
        results[mb] = "PASSED" if success else "FAILED"
        time.sleep(1)
        
    print("\n" + "="*50)
    print("           SWEEP TEST REPORT")
    print("="*50)
    for mb, res in results.items():
        print(f" Chunk Size: {mb:3d} MB  ->  Status: {res}")
    print("="*50)

if __name__ == "__main__":
    main()
