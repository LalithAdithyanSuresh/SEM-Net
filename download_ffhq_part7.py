import os
import sys
from huggingface_hub import snapshot_download

def download_part7(target_dir="datasets/ffhq_test_10k"):
    repo_id = "marcosv/ffhq-dataset"
    target_path = os.path.abspath(target_dir)
    
    print(f"[*] Downloading Part7 (10k test images) from Hugging Face: {repo_id}")
    print(f"[*] Saving destination: {target_path}\n")
    
    os.makedirs(target_path, exist_ok=True)
    
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["Part7/*"],
        local_dir=target_path,
        resume_download=True
    )
    
    print(f"\n[+] Download completed successfully!")
    print(f"[+] Part7 images stored at: {downloaded_path}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "datasets/ffhq_test_10k"
    download_part7(target)
