import os
import json
import time
import re
import shutil
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', static_url_path='')

# ── Multi-Session State ───────────────────────────────────────────────
# live_sessions[session_id] = { logs, metrics, state, last_seen }
live_sessions = {}

def get_session(session_id):
    # Sanitize session_id: remove whitespace, None, or empty
    if not session_id or str(session_id).lower() in ["none", "undefined", "null", ""]:
        session_id = "default"
    
    if session_id not in live_sessions:
        print(f"[NEW SESSION] Registered: {session_id}")
        live_sessions[session_id] = {
            "command": "stop",
            "shell_command": None,
            "logs_data": [],
            "global_log_count": 0,
            "psnr_values": [],
            "all_metrics_data": [],
            "last_seen": time.time()
        }
    else:
        live_sessions[session_id]["last_seen"] = time.time()
    return live_sessions[session_id]

# ── Disk paths ───────────────────────────────────────────────────────
UPLOAD_BASE = os.path.join(app.root_path, 'static', 'images')
os.makedirs(UPLOAD_BASE, exist_ok=True)

ARCHIVE_DIR = os.path.join(app.root_path, 'archive')
IMAGES_ARCHIVE_DIR = os.path.join(app.root_path, 'static', 'images_archive')
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(IMAGES_ARCHIVE_DIR, exist_ok=True)
ARCHIVE_FILE = os.path.join(ARCHIVE_DIR, 'runs.json')

def load_runs():
    if os.path.exists(ARCHIVE_FILE):
        try:
            with open(ARCHIVE_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_runs(runs):
    with open(ARCHIVE_FILE, 'w') as f: json.dump(runs, f)

# ── API Endpoints ─────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """Returns list of active live session IDs and archived runs."""
    active = []
    now = time.time()
    # Cleanup sessions older than 24 hours (optional) or just list all
    for sid, sdata in live_sessions.items():
        is_online = (now - sdata["last_seen"]) < 60 # Online if seen in last minute
        active.append({"id": sid, "status": "online" if is_online else "offline"})
    
    archived = list(load_runs().keys())
    return jsonify({"live": active, "archived": archived})

@app.route('/api/command', methods=['GET'])
def get_command():
    session_id = request.args.get('session', 'default')
    sess = get_session(session_id)
    return jsonify({"command": sess["command"], "shell_command": sess["shell_command"]})

@app.route('/api/pop_shell_command', methods=['GET'])
def pop_shell_command():
    session_id = request.args.get('session', 'default')
    sess = get_session(session_id)
    shell_cmd = sess.get('shell_command')
    sess['shell_command'] = None
    return jsonify({"shell_command": shell_cmd})

@app.route('/api/command', methods=['POST'])
def update_command():
    data = request.json
    session_id = data.get('session', 'default')
    sess = get_session(session_id)
    if 'command' in data:
        sess['command'] = data['command']
    if 'shell_command' in data:
        cmd_text = data['shell_command']
        sess['shell_command'] = cmd_text
        sess['logs_data'].append(f"> {cmd_text}")
    return jsonify({"status": "success"})

@app.route('/api/logs', methods=['POST'])
def add_logs():
    data = request.json
    session_id = data.get('session', 'default')
    sess = get_session(session_id)
    
    lines = data.get('lines', [])
    sess['logs_data'].extend(lines)
    sess['global_log_count'] += len(lines)

    # Keep buffer at 5000 lines
    if len(sess['logs_data']) > 5000:
        del sess['logs_data'][:-5000]

    for line in lines:
        match = re.search(r"psnr:\s*([\d\.]+)", line, re.IGNORECASE)
        if match:
            sess['psnr_values'].append(float(match.group(1)))
            if len(sess['psnr_values']) > 5000: sess['psnr_values'].pop(0)
    return jsonify({"status": "success"})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    run = request.args.get('run', '')
    session_id = request.args.get('session', '')
    after = request.args.get('after', type=int, default=-1)

    # Archive path
    if run:
        runs = load_runs()
        full_logs = runs.get(run, {}).get("logs_data", [])
        return jsonify({"lines": full_logs[after:] if after >= 0 else full_logs[-500:], "total": len(full_logs)})

    # Live path
    sess = get_session(session_id)
    if after >= 0:
        start_idx = sess['global_log_count'] - len(sess['logs_data'])
        if after < start_idx:
            return jsonify({"lines": sess['logs_data'], "total": sess['global_log_count']})
        rel_after = after - start_idx
        return jsonify({"lines": sess['logs_data'][rel_after:], "total": sess['global_log_count']})
    return jsonify({"lines": sess['logs_data'][-500:], "total": sess['global_log_count']})

@app.route('/api/all_metrics', methods=['POST'])
def add_metrics():
    data = request.json
    session_id = data.get('session', 'default')
    sess = get_session(session_id)
    sess['all_metrics_data'].append(data)
    if len(sess['all_metrics_data']) > 50000:
        del sess['all_metrics_data'][:-50000]
    return jsonify({"status": "success"})

@app.route('/api/all_metrics', methods=['GET'])
def get_metrics():
    run = request.args.get('run', '')
    session_id = request.args.get('session', '')
    after_iter = request.args.get('after', type=int, default=-1)

    if run:
        data = load_runs().get(run, {}).get("all_metrics_data", [])
    else:
        data = get_session(session_id)['all_metrics_data']

    if after_iter >= 0:
        data = [d for d in data if d.get('iteration', 0) > after_iter]
    return jsonify(data)

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    session_id = request.form.get('session', 'default')
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    
    # Isolate images by session in subfolders
    sess_dir = os.path.join(UPLOAD_BASE, session_id)
    os.makedirs(sess_dir, exist_ok=True)
    file.save(os.path.join(sess_dir, filename))
    return jsonify({"status": "success"})

@app.route('/api/images', methods=['GET'])
@app.route('/api/images/meta', methods=['GET'])
def get_images():
    run = request.args.get('run', '')
    session_id = request.args.get('session', 'default')
    is_meta = 'meta' in request.path

    target = os.path.join(IMAGES_ARCHIVE_DIR, run) if run else os.path.join(UPLOAD_BASE, session_id)
    if not os.path.exists(target): 
        return jsonify({"count": 0, "latest": None}) if is_meta else jsonify([])

    files = [f for f in os.listdir(target) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(target, x)))
    
    if is_meta:
        return jsonify({"count": len(files), "latest": files[-1] if files else None})
    return jsonify(files)

@app.route('/api/save_run', methods=['POST'])
def save_run():
    data = request.json
    run_name = data.get('name')
    session_id = data.get('session', 'default')
    if not run_name: return jsonify({"error": "No name"}), 400

    sess = get_session(session_id)
    runs = load_runs()
    runs[run_name] = {
        "logs_data": sess["logs_data"],
        "all_metrics_data": sess["all_metrics_data"],
        "psnr_values": sess["psnr_values"]
    }
    save_runs(runs)

    # Archive images
    src = os.path.join(UPLOAD_BASE, session_id)
    dst = os.path.join(IMAGES_ARCHIVE_DIR, run_name)
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        shutil.rmtree(src)

    # Clear live session
    del live_sessions[session_id]
    return jsonify({"status": "success"})

@app.route('/images/<session_id>/<filename>')
def serve_image(session_id, filename):
    return send_from_directory(os.path.join(UPLOAD_BASE, session_id), filename)

@app.route('/images_archive/<run_name>/<filename>')
def serve_archive_image(run_name, filename):
    return send_from_directory(os.path.join(IMAGES_ARCHIVE_DIR, run_name), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
