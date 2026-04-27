import os
import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import shutil
import re

app = Flask(__name__, static_folder='static', static_url_path='')

# ── In-memory state ──────────────────────────────────────────────────
state = {"command": "stop", "shell_command": None}

logs_data = []          # raw terminal lines (last 5000)
global_log_count = 0    # total lines ever added to logs_data
psnr_values = []        # legacy PSNR list (parsed from logs)
all_metrics_data = []   # structured dicts: every loss + psnr + mae per iteration

# ── Disk paths ───────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ARCHIVE_DIR       = os.path.join(app.root_path, 'archive')
IMAGES_ARCHIVE_DIR = os.path.join(app.root_path, 'static', 'images_archive')
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(IMAGES_ARCHIVE_DIR, exist_ok=True)
ARCHIVE_FILE = os.path.join(ARCHIVE_DIR, 'runs.json')

# ── Archive helpers ──────────────────────────────────────────────────
def load_runs():
    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_runs(runs):
    with open(ARCHIVE_FILE, 'w') as f:
        json.dump(runs, f)

# ── Static ────────────────────────────────────────────────────────────
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

# ── Training command endpoints ────────────────────────────────────────
@app.route('/api/command', methods=['GET'])
def get_command():
    return jsonify(state)

@app.route('/api/pop_shell_command', methods=['GET'])
def pop_shell_command():
    shell_cmd = state.get('shell_command')
    state['shell_command'] = None
    return jsonify({"shell_command": shell_cmd})

@app.route('/api/command', methods=['POST'])
def update_command():
    data = request.json
    if 'command' in data:
        state['command'] = data['command']
    if 'shell_command' in data:
        cmd_text = data['shell_command']
        state['shell_command'] = cmd_text
        logs_data.append(f"> {cmd_text}")
    return jsonify({"status": "success", "state": state})

# ── Archive / run management ──────────────────────────────────────────
@app.route('/api/runs', methods=['GET'])
def get_runs():
    return jsonify(list(load_runs().keys()))

@app.route('/api/save_run', methods=['POST'])
def save_run():
    global all_metrics_data, logs_data, psnr_values
    run_name = request.json.get('name')
    if not run_name:
        return jsonify({"error": "No name provided"}), 400

    runs = load_runs()
    runs[run_name] = {
        "logs_data": logs_data[-5000:],
        "psnr_values": psnr_values,
        "all_metrics_data": all_metrics_data[-50000:],  # keep last 50k entries
    }
    save_runs(runs)

    # Move images to archive
    run_image_dir = os.path.join(IMAGES_ARCHIVE_DIR, run_name)
    os.makedirs(run_image_dir, exist_ok=True)
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(os.path.join(UPLOAD_FOLDER, f), os.path.join(run_image_dir, f))

    # Reset live state
    all_metrics_data = []
    logs_data = []
    psnr_values = []
    return jsonify({"status": "success", "run": run_name})

# ── Logs ──────────────────────────────────────────────────────────────
@app.route('/api/logs', methods=['GET'])
def get_logs():
    run = request.args.get('run', '')
    after = request.args.get('after', type=int, default=-1)

    if run:
        runs = load_runs()
        full_logs = runs.get(run, {}).get("logs_data", [])
        if after >= 0:
            return jsonify(full_logs[after:])
        return jsonify(full_logs[-500:])

    # For live session, handle rotation
    if after >= 0:
        # Client asks for lines after its current count
        # global_log_count - len(logs_data) is the index of the first line currently in logs_data
        start_index_in_stream = global_log_count - len(logs_data)
        
        if after < start_index_in_stream:
            # Client is too far behind, send what we have
            return jsonify({"lines": logs_data, "total": global_log_count})
        
        relative_after = after - start_index_in_stream
        return jsonify({"lines": logs_data[relative_after:], "total": global_log_count})

    return jsonify({"lines": logs_data[-500:], "total": global_log_count})

@app.route('/api/logs', methods=['POST'])
def add_logs():
    global global_log_count
    data = request.json
    lines = data.get('lines', [])
    logs_data.extend(lines)
    global_log_count += len(lines)

    if len(logs_data) > 5000:
        del logs_data[:-5000]

    # Legacy PSNR parsing from raw log lines
    for line in lines:
        match = re.search(r"psnr:\s*([\d\.]+)", line, re.IGNORECASE)
        if match:
            psnr_values.append(float(match.group(1)))
            if len(psnr_values) > 5000:
                psnr_values.pop(0)
    return jsonify({"status": "success"})

# ── All structured metrics (NEW) ──────────────────────────────────────
@app.route('/api/all_metrics', methods=['POST'])
def add_all_metrics():
    """Receive one metrics snapshot dict per training iteration."""
    data = request.json
    if not data:
        return jsonify({"error": "empty body"}), 400

    iteration = data.get("iteration", 0)
    bucket_idx = iteration // 300

    if not all_metrics_data or all_metrics_data[-1].get('_bucket') != bucket_idx:
        data['_bucket'] = bucket_idx
        data['_count'] = 1
        all_metrics_data.append(data)
    else:
        b = all_metrics_data[-1]
        c = b['_count']
        b['_count'] = c + 1
        for k, v in data.items():
            if isinstance(v, (int, float)) and k not in ['iteration', 'epoch', '_bucket', '_count']:
                b[k] = (b[k] * c + v) / (c + 1)
        b['iteration'] = iteration
        b['epoch'] = data.get('epoch', b['epoch'])

    # Trim: 50k buckets (15M iterations) is plenty
    if len(all_metrics_data) > 50000:
        del all_metrics_data[:-50000]
    return jsonify({"status": "success"})

@app.route('/api/all_metrics', methods=['GET'])
def get_all_metrics():
    """Return all structured metrics for the current session or an archived run."""
    run = request.args.get('run', '')
    after_iter = request.args.get('after', type=int, default=-1)

    if run:
        runs = load_runs()
        data = runs.get(run, {}).get("all_metrics_data", [])
    else:
        data = all_metrics_data

    # Delta fetch limit
    if after_iter >= 0:
        data = [d for d in data if d.get('iteration', 0) > after_iter]

    return jsonify(data)

# ── Legacy metrics endpoint (kept for compat) ─────────────────────────
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    run = request.args.get('run', '')
    if run:
        runs = load_runs()
        archived = runs.get(run, {})
        return jsonify(archived.get("psnr_values", []))
    return jsonify(psnr_values)

@app.route('/api/metrics', methods=['POST'])
def add_metrics_legacy():
    # kept for any external tools still posting here
    data = request.json
    if data:
        psnr_values.append(data)
        if len(psnr_values) > 5000:
            psnr_values.pop(0)
    return jsonify({"status": "success"})

# ── Image endpoints ───────────────────────────────────────────────────
@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({"status": "success", "filename": filename})

@app.route('/api/images/meta', methods=['GET'])
def images_meta():
    run = request.args.get('run', '')
    target_folder = os.path.join(IMAGES_ARCHIVE_DIR, run) if run else UPLOAD_FOLDER
    if os.path.exists(target_folder):
        files = [f for f in os.listdir(target_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(target_folder, x)))
        return jsonify({"count": len(files), "latest": files[-1] if files else None})
    return jsonify({"count": 0, "latest": None})

@app.route('/api/images', methods=['GET'])
def list_images():
    run = request.args.get('run', '')
    target_folder = os.path.join(IMAGES_ARCHIVE_DIR, run) if run else UPLOAD_FOLDER
    images = []
    if os.path.exists(target_folder):
        all_files = [f for f in os.listdir(target_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        all_files.sort(key=lambda x: os.path.getmtime(os.path.join(target_folder, x)))
        images = all_files
    return jsonify(images)

@app.route('/api/sync_result', methods=['POST'])
def sync_result():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    data_dir = os.path.expanduser('~/data')
    os.makedirs(data_dir, exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = os.path.join(data_dir, filename)
    file.save(save_path)
    return jsonify({"status": "success", "path": save_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
