#!/usr/bin/env python3
import os
import sys
import argparse
import requests
import time

def upload_file_chunked(file_path, server_url, session_id, chunk_size=10 * 1024 * 1024):
    if not os.path.exists(file_path):
        print(f"[ERROR] File {file_path} not found.")
        return False
        
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    
    print(f"[UPLOAD] Uploading {filename} ({file_size / (1024*1024):.2f} MB) in {total_chunks} chunks...")
    
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
                        res = requests.post(f"{server_url}/api/upload_chunk", files=files, data=data, timeout=60)
                        if res.status_code == 200:
                            success = True
                            break
                        else:
                            print(f"  -> Chunk {i} attempt {retry+1} failed with status {res.status_code}: {res.text[:200]}")
                    except Exception as e:
                        print(f"  -> Chunk {i} attempt {retry+1} error: {e}")
                    time.sleep(1)
                    
                if not success:
                    print(f"[ERROR] Failed to upload chunk {i}.")
                    return False
        print(f"[SUCCESS] Successfully uploaded {filename} to {server_url}/download/{filename}")
        return True
    except Exception as e:
        print(f"[ERROR] Error uploading {filename}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload any file to the files server.")
    parser.add_argument("file_path", help="Path to the file to upload")
    parser.add_argument("--server", default="https://files.lalithadithyan.dev", help="Files server base URL")
    parser.add_argument("--session", default="DAVA", help="Session ID")
    parser.add_argument("--chunk-size", type=int, default=10, help="Chunk size in MB (default: 10)")
    
    args = parser.parse_args()
    
    chunk_bytes = args.chunk_size * 1024 * 1024
    upload_file_chunked(args.file_path, args.server, args.session, chunk_bytes)

if __name__ == "__main__":
    main()
