import sys
import requests
import os
import time
import re

C2_SERVER_URL = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')
C2_SESSION    = os.environ.get('C2_SESSION', 'DAVA')

# Set up local log file that only records until iteration 100
local_log_file = f"training_{C2_SESSION}.log"
log_fp = None
try:
    log_fp = open(local_log_file, "a")
except Exception:
    pass
stop_local_logging = False
iteration_counter = 0

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
            
            # Local logging until iteration >= 100
            if log_fp and not stop_local_logging:
                try:
                    log_fp.write(line + "\n")
                    log_fp.flush()
                    # Check for iteration progress bar (e.g., "1388/450865 [")
                    if re.search(r'^\d+/\d+\s+\[', line):
                        iteration_counter += 1
                        if iteration_counter >= 100:
                            stop_local_logging = True
                            log_fp.write("--- STOPPING LOCAL LOGGING AFTER 100 ITERATIONS ---\n")
                            log_fp.flush()
                            log_fp.close()
                            log_fp = None
                except Exception:
                    pass

        current_line = ""
        
        # Flush buffer locally without network POST
        if len(buffer) >= 50 or (time.time() - last_push) > 2.0:
            buffer = []
            last_push = time.time()
    else:
        current_line += char

# Clear remaining buffer without network POST
if buffer or current_line:
    if current_line:
        buffer.append(current_line.strip())
        if log_fp and not stop_local_logging:
            try:
                log_fp.write(current_line.strip() + "\n")
                log_fp.flush()
            except Exception:
                pass

if log_fp:
    try:
        log_fp.close()
    except Exception:
        pass
