import urllib.request
import os
import time

url = "https://files.lalithadithyan.dev/download/CCC.zip"
output_path = "CCC.zip"

print(f"Downloading {url} to {output_path}...")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for attempt in range(5):
    try:
        # Check if file partially exists to support resume (if server supports range headers)
        # For simplicity, let's just do a clean chunked download first, but if it aborts, we retry.
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            meta = response.info()
            file_size = meta.get("Content-Length")
            if file_size:
                file_size = int(file_size)
                print(f"Total file size: {file_size / (1024*1024):.2f} MB")
            else:
                print("Total file size: Unknown")
                
            chunk_size = 1024 * 1024 # 1MB chunks
            bytes_downloaded = 0
            
            with open(output_path, "wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    bytes_downloaded += len(chunk)
                    if file_size:
                        percent = (bytes_downloaded / file_size) * 100
                        print(f"Downloaded {bytes_downloaded / (1024*1024):.2f} MB ({percent:.1f}%)", end="\r")
                    else:
                        print(f"Downloaded {bytes_downloaded / (1024*1024):.2f} MB", end="\r")
            
            print("\nDownload completed successfully!")
            break
    except Exception as e:
        print(f"\nAttempt {attempt+1} failed with error: {e}")
        time.sleep(2)
