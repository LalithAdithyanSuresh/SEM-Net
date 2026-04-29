import os
import glob
import pandas as pd
pd.options.compute.use_bottleneck = False
import json
import sqlite3
import zipfile
import shutil
import io
from PIL import Image
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, render_template, request, send_from_directory, send_file

app = Flask(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
os.makedirs(DATA_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect('validation_results.db')
    c = conn.cursor()
    
    # Check if old table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='votes'")
    if c.fetchone():
        # Check if the table has model1 column
        c.execute("PRAGMA table_info(votes)")
        columns = [col[1] for col in c.fetchall()]
        if 'model1' not in columns:
            print("Migrating database...")
            c.execute("ALTER TABLE votes RENAME TO votes_old")
            c.execute('''CREATE TABLE votes
                         (image_id TEXT, size TEXT, model1 TEXT, model2 TEXT, winner TEXT, comment TEXT, PRIMARY KEY(image_id, size, model1, model2))''')
            c.execute('''INSERT INTO votes (image_id, size, model1, model2, winner, comment)
                         SELECT image_id, size, 'evaluation_results_standard_uniform', 'deterministic_strided', winner, comment FROM votes_old''')
            conn.commit()
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS votes
                     (image_id TEXT, size TEXT, model1 TEXT, model2 TEXT, winner TEXT, comment TEXT, PRIMARY KEY(image_id, size, model1, model2))''')
        conn.commit()
    
    # Ensure updated_at column exists
    c.execute("PRAGMA table_info(votes)")
    columns = [col[1] for col in c.fetchall()]
    if 'updated_at' not in columns:
        print("Adding updated_at column...")
        c.execute("ALTER TABLE votes ADD COLUMN updated_at INTEGER DEFAULT 0")
        conn.commit()
        
    conn.close()

init_db()

def get_grid_files(folder, size):
    grid_dir = os.path.join(folder, '5_image_grid', size)
    if not os.path.exists(grid_dir):
        return {}
    files = os.listdir(grid_dir)
    mapping = {}
    for f in files:
        # Format is usually SIZE_ID_PSNR.png e.g. SMALL_00000_44.30.png
        parts = f.split('_')
        if len(parts) >= 2:
            img_id = parts[1]
            mapping[img_id] = f
    return mapping

@app.route('/api/folders')
def api_folders():
    folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    return jsonify(folders)

@app.route('/api/data')
def api_data():
    size = request.args.get('size', 'SMALL')
    model1 = request.args.get('model1')
    model2 = request.args.get('model2')
    
    if not model1 or not model2:
        return jsonify([])
        
    m1_path = os.path.join(DATA_DIR, model1, f'metrics_{size}.csv')
    m2_path = os.path.join(DATA_DIR, model2, f'metrics_{size}.csv')
    
    df1 = pd.read_csv(m1_path, on_bad_lines='skip') if os.path.exists(m1_path) else pd.DataFrame(columns=['Image', 'PSNR'])
    df2 = pd.read_csv(m2_path, on_bad_lines='skip') if os.path.exists(m2_path) else pd.DataFrame(columns=['Image', 'PSNR'])
    
    if 'Image' not in df1.columns: df1['Image'] = []
    if 'Image' not in df2.columns: df2['Image'] = []
    if 'PSNR' not in df1.columns: df1['PSNR'] = 0
    if 'PSNR' not in df2.columns: df2['PSNR'] = 0
    
    df1 = df1.rename(columns={'PSNR': 'PSNR_1'})
    df2 = df2.rename(columns={'PSNR': 'PSNR_2'})
    
    merged = pd.merge(df1[['Image', 'PSNR_1']], df2[['Image', 'PSNR_2']], on='Image', how='outer')
    merged = merged.fillna(0)
    
    grid1 = get_grid_files(os.path.join(DATA_DIR, model1), size)
    grid2 = get_grid_files(os.path.join(DATA_DIR, model2), size)
    
    conn = sqlite3.connect('validation_results.db')
    c = conn.cursor()
    c.execute('SELECT image_id, winner, comment FROM votes WHERE size=? AND model1=? AND model2=?', (size, model1, model2))
    votes_rows = c.fetchall()
    conn.close()
    
    votes_dict = {row[0]: {'winner': row[1], 'comment': row[2]} for row in votes_rows}
    
    results = []
    for _, row in merged.iterrows():
        image_file = row['Image']
        img_id = image_file.replace('.jpg', '').replace('.png', '')
        
        vote_data = votes_dict.get(img_id, {'winner': None, 'comment': ''})
        
        results.append({
            'id': img_id,
            'filename': image_file,
            'psnr_1': round(float(row['PSNR_1']), 2),
            'psnr_2': round(float(row['PSNR_2']), 2),
            'f1_fake': f'/images/{model1}/fid_fake_{size}/{image_file}',
            'f2_fake': f'/images/{model2}/fid_fake_{size}/{image_file}',
            'gt': f'/images/{model1}/fid_real_{size}/{image_file}',
            'f1_grid': f'/images/{model1}/5_image_grid/{size}/{grid1.get(img_id, "")}' if grid1.get(img_id) else "",
            'f2_grid': f'/images/{model2}/5_image_grid/{size}/{grid2.get(img_id, "")}' if grid2.get(img_id) else "",
            'winner': vote_data['winner'],
            'comment': vote_data['comment']
        })
    
    results.sort(key=lambda x: x['id'])
    return jsonify(results)

@app.route('/api/votes_sync')
def api_votes_sync():
    size = request.args.get('size', 'SMALL')
    model1 = request.args.get('model1')
    model2 = request.args.get('model2')
    since = int(request.args.get('since', 0))
    
    if not model1 or not model2:
        return jsonify({})
        
    conn = sqlite3.connect('validation_results.db')
    c = conn.cursor()
    c.execute('SELECT image_id, winner, comment, updated_at FROM votes WHERE size=? AND model1=? AND model2=? AND updated_at > ?', 
              (size, model1, model2, since))
    votes_rows = c.fetchall()
    conn.close()
    
    votes_dict = {row[0]: {'winner': row[1], 'comment': row[2], 'updated_at': row[3]} for row in votes_rows}
    return jsonify(votes_dict)

@app.route('/api/vote', methods=['POST'])
def api_vote():
    data = request.json
    image_id = data.get('image_id')
    size = data.get('size')
    model1 = data.get('model1')
    model2 = data.get('model2')
    winner = data.get('winner')
    comment = data.get('comment', '')
    
    import time
    updated_at = int(time.time())
    
    if not image_id or not size or not model1 or not model2:
        return jsonify({'error': 'Missing required fields'}), 400
        
    conn = sqlite3.connect('validation_results.db')
    c = conn.cursor()
    c.execute('''INSERT INTO votes (image_id, size, model1, model2, winner, comment, updated_at) 
                 VALUES (?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(image_id, size, model1, model2) 
                 DO UPDATE SET winner=excluded.winner, comment=excluded.comment, updated_at=excluded.updated_at''',
              (image_id, size, model1, model2, winner, comment, updated_at))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/upload_chunk', methods=['POST'])
def api_upload_chunk():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    filename = secure_filename(request.form.get('filename', ''))
    
    try:
        chunk_index = int(request.form.get('chunk_index', -1))
        total_chunks = int(request.form.get('total_chunks', -1))
    except ValueError:
        return jsonify({'error': 'Invalid chunk metadata'}), 400
    
    if not filename or chunk_index < 0 or total_chunks <= 0:
        return jsonify({'error': 'Missing metadata'}), 400
        
    temp_dir = os.path.join(DATA_DIR, f"temp_{filename}")
    os.makedirs(temp_dir, exist_ok=True)
    
    chunk_path = os.path.join(temp_dir, str(chunk_index))
    file.save(chunk_path)
    
    # Check if all chunks have been received
    downloaded_chunks = len(os.listdir(temp_dir))
    if downloaded_chunks == total_chunks:
        final_zip_path = os.path.join(DATA_DIR, filename)
        try:
            with open(final_zip_path, 'wb') as outfile:
                for i in range(total_chunks):
                    cp = os.path.join(temp_dir, str(i))
                    with open(cp, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
            
            # Clean up chunks
            shutil.rmtree(temp_dir)
            
            # Extract
            extract_dir = os.path.join(DATA_DIR, filename.replace('.zip', ''))
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(final_zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
                
            os.remove(final_zip_path)
            return jsonify({'success': True, 'message': 'Upload and extraction complete'})
            
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(final_zip_path):
                os.remove(final_zip_path)
            return jsonify({'error': f'Assembly failed: {str(e)}'}), 500
            
    return jsonify({'success': True, 'message': f'Chunk {chunk_index} received'})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.zip'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(DATA_DIR, filename)
        file.save(filepath)
        
        # Extract folder name from zip (assume root folder name)
        extract_dir = os.path.join(DATA_DIR, filename.replace('.zip', ''))
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                # To prevent creating nested folders if zip already contains root folder,
                # we just extract it to DATA_DIR directly, or extract to extract_dir.
                zip_ref.extractall(DATA_DIR)
            os.remove(filepath)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/images/<model_name>/<path:filename>')
def serve_image(model_name, filename):
    model_dir = os.path.join(DATA_DIR, model_name)
    return send_from_directory(model_dir, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/mask_only/<model>/<size>/<image_name>')
def api_mask_only(model, size, image_name):
    model_dir = os.path.join(DATA_DIR, model)
    grid_mapping = get_grid_files(model_dir, size)
    
    grid_file = grid_mapping.get(image_name)
    
    if not grid_file:
        return f"Grid not found for ID {image_name}", 404
        
    grid_path = os.path.join(model_dir, '5_image_grid', size, grid_file)
    try:
        with Image.open(grid_path) as img:
            W, H = img.size
            w = W // 5 # Assume 5-image layout
            # Crop the 2nd image (masked input)
            mask_crop = img.crop((w, 0, 2*w, H))
            
            # Save to buffer
            buf = io.BytesIO()
            mask_crop.save(buf, format='PNG', optimize=True)
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
