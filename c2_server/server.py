import os
import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', static_url_path='')

# Initial state
state = {
    "command": "stop",
    "shell_command": None
}

metrics_data = []
logs_data = []

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/api/command', methods=['GET'])
def get_command():
    # Regular polling is now read-only (won't clear shell_command)
    return jsonify(state)

@app.route('/api/pop_shell_command', methods=['GET'])
def pop_shell_command():
    # Dedicated endpoint for the background script to safely 'consume' the command
    shell_cmd = state.get('shell_command')
    state['shell_command'] = None # Clear it now that it's being consumed
    return jsonify({"shell_command": shell_cmd})

@app.route('/api/command', methods=['POST'])
def update_command():
    data = request.json
    if 'command' in data:
        state['command'] = data['command']
    if 'shell_command' in data:
        cmd_text = data['shell_command']
        state['shell_command'] = cmd_text
        # Add the command to the logs so it appears in the live terminal
        logs_data.append(f"> {cmd_text}")
    return jsonify({"status": "success", "state": state})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics_data)

@app.route('/api/metrics', methods=['POST'])
def add_metrics():
    data = request.json
    metrics_data.append(data)
    # limit history to last 1000 epochs
    if len(metrics_data) > 1000:
        metrics_data.pop(0)
    return jsonify({"status": "success"})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    # Return last 500 lines to avoid massive payload
    return jsonify(logs_data[-500:])

@app.route('/api/logs', methods=['POST'])
def add_logs():
    data = request.json
    lines = data.get('lines', [])
    logs_data.extend(lines)
    if len(logs_data) > 5000:
        del logs_data[:-5000]
    return jsonify({"status": "success"})

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"status": "success", "filename": filename})

@app.route('/api/images', methods=['GET'])
def list_images():
    images = []
    if os.path.exists(UPLOAD_FOLDER):
        images = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
        images.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
    return jsonify(images)

@app.route('/api/sync_result', methods=['POST'])
def sync_result():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save to ~/data
    data_dir = os.path.expanduser('~/data')
    os.makedirs(data_dir, exist_ok=True)
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(data_dir, filename)
    file.save(save_path)
    
    return jsonify({"status": "success", "path": save_path})

if __name__ == '__main__':
    # When deployed with Gunicorn, this block is bypassed.
    app.run(host='0.0.0.0', port=5000)
