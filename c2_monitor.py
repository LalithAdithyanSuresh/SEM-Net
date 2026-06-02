import argparse
import os
import time
import requests
import subprocess
import sys

# Configure these or pass via environment variables
C2_SERVER_URL = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')
C2_SESSION = os.environ.get('C2_SESSION', 'default')

def get_current_epoch():
    """
    Fetches the latest metrics from the C2 server to determine the current epoch.
    """
    try:
        # We query the status endpoint which you need to add to your C2 Server
        res = requests.get(f"{C2_SERVER_URL}/api/status", params={"session": C2_SESSION}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return data.get('epoch', -1)
    except Exception as e:
        print(f"[ERROR] Fetching status: {e}")
    return -1

def send_stop_command():
    """
    Sends the 'stop' command to the C2 server so the training script will pick it up and halt.
    """
    try:
        # Assuming you will add an endpoint /api/set_command to your C2 server
        res = requests.post(f"{C2_SERVER_URL}/api/set_command", 
                            json={"session": C2_SESSION, "command": "stop"}, 
                            timeout=5)
        if res.status_code == 200:
            print("[SUCCESS] Sent STOP command to C2 server.")
            return True
    except Exception as e:
        print(f"[ERROR] Sending STOP command: {e}")
    return False

def push_logs(lines):
    """
    Pushes an array of log lines to the C2 dashboard terminal.
    """
    try:
        requests.post(f"{C2_SERVER_URL}/api/logs", 
                      json={"lines": lines, "session": C2_SESSION}, 
                      timeout=2)
    except Exception:
        pass

def stream_output_to_c2(process):
    """
    Reads the stdout/stderr of a running subprocess character by character,
    prints it locally, and buffers it to push to the C2 server every 2s or 50 lines.
    """
    buffer = []
    last_push = time.time()
    current_line = ""
    
    while True:
        char = process.stdout.read(1)
        
        if not char:
            if process.poll() is not None:
                break
            time.sleep(0.01)
            continue
            
        sys.stdout.write(char)
        sys.stdout.flush()
        
        if char == '\r' or char == '\n':
            line = current_line.strip()
            if line:
                buffer.append(line)
            current_line = ""
            
            # Push every 2 seconds or 50 lines to keep it live without spamming
            if len(buffer) >= 50 or (time.time() - last_push) > 2.0:
                if buffer:
                    push_logs(buffer)
                    buffer = []
                last_push = time.time()
        else:
            current_line += char

    # Push remaining lines when process exits
    if buffer or current_line:
        if current_line:
            buffer.append(current_line.strip())
        push_logs(buffer)

def main():
    parser = argparse.ArgumentParser(description="Monitor C2 Server and trigger a local command once a target epoch is reached.")
    parser.add_argument('--target-epoch', type=float, required=True, help="Epoch to stop training at")
    parser.add_argument('--command', type=str, required=True, help="Command to run after stopping (e.g., 'python evaluate.py')")
    parser.add_argument('--poll-interval', type=int, default=10, help="Seconds between polling C2 server")
    args = parser.parse_args()

    print(f"[*] Monitoring C2 server ({C2_SERVER_URL}) for session '{C2_SESSION}'")
    print(f"[*] Waiting for epoch to reach >= {args.target_epoch}")

    while True:
        current_epoch = get_current_epoch()
        if current_epoch != -1:
            print(f"Current Epoch: {current_epoch} / Target: {args.target_epoch}")
            if current_epoch >= args.target_epoch:
                print(f"\n[!] Target epoch {args.target_epoch} reached! (Current: {current_epoch})")
                break
        else:
            print("Could not get current epoch, retrying...")
            
        time.sleep(args.poll_interval)

    print("\n[*] Sending STOP command to C2 server...")
    success = send_stop_command()
    
    if not success:
        print("[!] Warning: Could not send STOP command. Proceeding to run the post-script anyway.")

    # Give the training script time to receive the stop command and gracefully shut down
    # so we don't hit Out Of Memory errors when running the post-training evaluation script
    print("\n[*] Waiting 15 seconds for the training process to gracefully halt...")
    time.sleep(15)

    print(f"\n[*] Executing post-training command: {args.command}")
    push_logs([
        f"==========================================",
        f"--- TRIGGERED POST-TRAINING SCRIPT ---", 
        f"Command: {args.command}",
        f"=========================================="
    ])
    
    # Run the user's specific code
    process = subprocess.Popen(
        args.command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1 # Line buffered
    )

    # Stream the live output to C2
    stream_output_to_c2(process)
    
    process.wait()
    print(f"\n[*] Command finished with exit code {process.returncode}")
    push_logs([f"--- COMMAND FINISHED (Exit code {process.returncode}) ---"])

if __name__ == "__main__":
    main()
