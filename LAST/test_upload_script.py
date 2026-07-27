import os
import requests
import time

def upload_file_chunked(file_path, server_url, session_id, chunk_size=10 * 1024 * 1024):
    if not os.path.exists(file_path):
        print(f"[TEST UPLOAD] File {file_path} not found.")
        return False
        
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    
    print(f"[TEST UPLOAD] Uploading {filename} ({file_size / (1024*1024):.2f} MB) in {total_chunks} chunks...")
    print(f"[TEST UPLOAD] Destination URL: {server_url}/api/upload_chunk")
    
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
                
                success = False
                for retry in range(3):
                    try:
                        res = requests.post(f"{server_url}/api/upload_chunk", files=files, data=data, timeout=45)
                        print(f"  -> Chunk {i}: Server returned status {res.status_code}")
                        if res.status_code == 200:
                            success = True
                            break
                        else:
                            print(f"  -> Response content: {res.text[:300]}")
                    except Exception as e:
                        print(f"  -> Chunk {i} retry {retry+1} error: {e}")
                    time.sleep(1)
                    
                if not success:
                    print(f"[TEST UPLOAD] Failed to upload chunk {i}.")
                    return False
        print(f"[TEST UPLOAD] Successfully uploaded {filename}!")
        return True
    except Exception as e:
        print(f"[TEST UPLOAD] Error uploading {filename}: {e}")
        return False

def main():
    test_file = "test_upload_file.bin"
    # Create a dummy 100KB file
    print("[TEST UPLOAD] Creating a 100KB dummy file...")
    with open(test_file, "wb") as f:
        f.write(os.urandom(100 * 1024))
    
    server_url = "https://files.lalithadithyan.dev"
    session_id = "test_session_laptop"
    
    print(f"[TEST UPLOAD] Target Server: {server_url}")
    print(f"[TEST UPLOAD] Session: {session_id}")
    
    # Use 50KB chunk size
    success = upload_file_chunked(test_file, server_url, session_id, chunk_size=50 * 1024)
    
    # Clean up test file
    if os.path.exists(test_file):
        os.remove(test_file)
        
    if success:
        print("[TEST UPLOAD] Test PASSED successfully!")
    else:
        print("[TEST UPLOAD] Test FAILED!")

if __name__ == "__main__":
    main()
