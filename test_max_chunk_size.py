import os
import requests
import time

def measure_speed(server_url, session_id):
    # Create 1MB dummy file
    filename = "speed_test.bin"
    with open(filename, "wb") as f:
        f.write(os.urandom(1 * 1024 * 1024))
        
    print("[SPEED TEST] Measuring upload speed with a 1MB file...")
    
    data = {
        'session': session_id,
        'filename': filename,
        'chunk_index': 0,
        'total_chunks': 1
    }
    
    start = time.time()
    try:
        with open(filename, "rb") as fileobj:
            files = {'file': (f"{filename}.part0", fileobj, 'application/octet-stream')}
            res = requests.post(f"{server_url}/api/upload_chunk", files=files, data=data, timeout=60)
        
        elapsed = time.time() - start
        if res.status_code == 200:
            speed_mbps = (8.0) / elapsed  # Megabits per second
            speed_mbs = 1.0 / elapsed     # Megabytes per second
            print(f"[SPEED TEST] Completed. Speed: {speed_mbs:.2f} MB/s ({speed_mbps:.2f} Mbps). Elapsed: {elapsed:.2f}s")
            return speed_mbs
        else:
            print(f"[SPEED TEST] Failed with code {res.status_code}: {res.text[:100]}")
            return None
    except Exception as e:
        print(f"[SPEED TEST] Error: {e}")
        return None
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def run_upload_test(file_path, chunk_size, server_url, session_id, timeout_sec):
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    
    print(f"\n[TEST] Uploading {filename} ({file_size / (1024*1024):.2f} MB) in {total_chunks} chunks (size: {chunk_size / (1024*1024):.2f} MB) with timeout: {timeout_sec:.1f}s...")
    
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
                    print(f"   -> Chunk {i+1}/{total_chunks} uploaded successfully in {chunk_elapsed:.2f}s")
                else:
                    print(f"   -> Chunk {i+1}/{total_chunks} failed. Status: {res.status_code}, Response: {res.text[:200]}")
                    return False
        
        total_elapsed = time.time() - start_time
        print(f"[SUCCESS] Uploaded {filename} in {total_elapsed:.2f}s!")
        return True
    except Exception as e:
        print(f"[FAILED] Error during upload: {e}")
        return False

def main():
    server_url = "https://files.lalithadithyan.dev"
    session_id = "chunk_sweep_test"
    
    # 1. Measure speed
    speed = measure_speed(server_url, session_id)
    if not speed:
        print("[ERROR] Speed test failed. Cannot calculate safe timeouts.")
        speed = 0.5  # fallback assumption of 500 KB/s
    
    # 2. Define test chunk sizes (in Megabytes)
    test_sizes_mb = [5, 10, 25, 50]
    
    results = {}
    
    for mb in test_sizes_mb:
        chunk_size_bytes = mb * 1024 * 1024
        # Calculate dynamic timeout: time to upload chunk + 20 seconds padding
        dynamic_timeout = (mb / speed) * 3 + 20
        dynamic_timeout = max(30, min(dynamic_timeout, 300))  # limit timeout between 30s and 5m
        
        # File size is twice the chunk size (so it splits into 2 chunks)
        test_file = f"test_{mb}mb.bin"
        file_size_bytes = chunk_size_bytes * 2
        
        print(f"\n" + "="*60)
        print(f"Testing Chunk Size: {mb} MB (File size: {mb*2} MB)")
        print(f"Calculated Dynamic Timeout: {dynamic_timeout:.1f} seconds")
        print(f"="*60)
        
        # Create temporary dummy file
        with open(test_file, "wb") as f:
            f.write(os.urandom(file_size_bytes))
            
        success = run_upload_test(test_file, chunk_size_bytes, server_url, session_id, dynamic_timeout)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
            
        results[mb] = "PASSED" if success else "FAILED"
        
        # Small delay between runs
        time.sleep(2)
        
    print("\n" + "="*40)
    print("           SWEEP TEST REPORT")
    print("="*40)
    for mb, res in results.items():
        print(f" Chunk Size: {mb:2d} MB  ->  Status: {res}")
    print("="*40)

if __name__ == "__main__":
    main()
