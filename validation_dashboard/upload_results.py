import os
import sys
import shutil
import requests
import argparse

def upload_folder(folder_path, host):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return
        
    folder_name = os.path.basename(os.path.normpath(folder_path))
    zip_path = f"{folder_name}.zip"
    
    print(f"Zipping {folder_path} to {zip_path}...")
    shutil.make_archive(folder_name, 'zip', folder_path)
    
    upload_url = f"{host.rstrip('/')}/api/upload"
    
    print(f"Uploading {zip_path} to {upload_url}...")
    try:
        with open(zip_path, 'rb') as f:
            files = {'file': (zip_path, f, 'application/zip')}
            response = requests.post(upload_url, files=files)
            
        if response.status_code == 200:
            print("Upload successful!")
        else:
            print(f"Upload failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up temporary zip file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload validation results to the dashboard server.")
    parser.add_argument("folder", help="Path to the folder containing results (e.g. inference_results/deterministic_strided)")
    parser.add_argument("--host", default="http://localhost:5003", help="URL of the dashboard server (e.g. https://validate.lalithadithyan.dev)")
    
    args = parser.parse_args()
    upload_folder(args.folder, args.host)
