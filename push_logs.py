import sys
import requests
import os
import time

C2_SERVER_URL = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')
buffer = []
last_push = time.time()

for line in sys.stdin:
    sys.stdout.write(line)
    sys.stdout.flush()
    buffer.append(line.strip())
    
    # Push every 2 seconds or 50 lines
    if len(buffer) >= 50 or (time.time() - last_push) > 2.0:
        if buffer:
            try:
                requests.post(f"{C2_SERVER_URL}/api/logs", json={"lines": buffer}, timeout=2)
            except Exception:
                pass
            buffer = []
        last_push = time.time()

# Push remaining
if buffer:
    try:
        requests.post(f"{C2_SERVER_URL}/api/logs", json={"lines": buffer}, timeout=2)
    except Exception:
        pass
