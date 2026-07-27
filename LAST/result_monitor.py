import os
import time
import requests

# Use the same C2 URL as the training script
C2_SERVER_URL = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')
RESULTS_DIR = './checkpoints_c2/results/inpaint/validation'

SENT_LOG = '.sent_results'

def get_sent_files():
    if not os.path.exists(SENT_LOG):
        return set()
    with open(SENT_LOG, 'r') as f:
        return set(line.strip() for line in f)

def mark_as_sent(filename):
    with open(SENT_LOG, 'a') as f:
        f.write(filename + '\n')

def main():
    print(f"Monitoring {RESULTS_DIR} for new results...")
    print(f"Server: {C2_SERVER_URL}")
    sent_files = get_sent_files()
    
    while True:
        try:
            if not os.path.exists(RESULTS_DIR):
                time.sleep(10)
                continue
                
            current_files = os.listdir(RESULTS_DIR)
            for f in current_files:
                # Sync only if it's a file we haven't sent yet
                file_path = os.path.join(RESULTS_DIR, f)
                if f not in sent_files and os.path.isfile(file_path):
                    print(f"Syncing new result: {f}")
                    try:
                        with open(file_path, 'rb') as img:
                            files = {'file': (f, img)}
                            res = requests.post(f"{C2_SERVER_URL}/api/sync_result", files=files, timeout=60)
                            if res.status_code == 200:
                                print(f"Successfully synced {f}")
                                sent_files.add(f)
                                mark_as_sent(f)
                            else:
                                print(f"Failed to sync {f}: {res.text}")
                    except Exception as pulse_err:
                        print(f"Error during POST for {f}: {pulse_err}")
            
        except Exception as e:
            print(f"Monitor Loop Error: {e}")
            
        time.sleep(15) # Poll every 15 seconds for new images

if __name__ == '__main__':
    main()
