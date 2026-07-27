import zipfile
import os

zip_path = "CCC.zip"
extract_path = "CCC"

print(f"Extracting {zip_path} to {extract_path}...")
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed successfully!")
    
    # List top level contents
    items = os.listdir(extract_path)
    print(f"Total items in extracted folder: {len(items)}")
    print("Sample items:")
    for i in items[:10]:
        item_path = os.path.join(extract_path, i)
        if os.path.isdir(item_path):
            print(f"  [DIR]  {i} (contains {len(os.listdir(item_path))} items)")
        else:
            print(f"  [FILE] {i}")
except Exception as e:
    print(f"Error extracting zip: {e}")
