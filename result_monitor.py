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
    print("Network result sync is disabled.")
    return
