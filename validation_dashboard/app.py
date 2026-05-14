import os
import glob
import pandas as pd
pd.options.compute.use_bottleneck = False
import json
import sqlite3
import zipfile
import shutil
import io
import numpy as np
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

import json

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'mask_cache.json')

def get_indexed_masks():
    global indexed_masks_cache
    if indexed_masks_cache:
        return indexed_masks_cache
        
    mask_dir = os.path.join(DATA_DIR, 'testing_mask_dataset')
    if not os.path.exists(mask_dir):
        return {'SMALL': [], 'MEDIUM': [], 'LARGE': []}
        
    mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    mask_files.sort() # Critical for deterministic mapping
    
    # 4k / 4k / 4k split as specified by user
    # Handle cases where count might not be exactly 12k
    n = len(mask_files)
    s1 = min(n, 4000)
    s2 = min(n, 8000)
    
    categories = {
        'SMALL': mask_files[0:s1],
        'MEDIUM': mask_files[s1:s2],
        'LARGE': mask_files[s2:]
    }
                
    indexed_masks_cache = categories
    print(f"Indexed {n} masks with 4k split: {len(categories['SMALL'])} SMALL, {len(categories['MEDIUM'])} MEDIUM, {len(categories['LARGE'])} LARGE")
    return categories

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
    # Exclude mask datasets from model selection
    excluded = ['testing_mask_dataset', 'masks', 'iregularmask']
    folders = [f for f in os.listdir(DATA_DIR) 
               if os.path.isdir(os.path.join(DATA_DIR, f)) and f not in excluded]
    return jsonify(folders)

@app.route('/api/data')
def api_data():
    size = request.args.get('size', 'SMALL')
    models = [request.args.get(f'model{i}') for i in range(1, 6)]
    models = [m for m in models if m] # filter out empty ones
    
    if not models:
        return jsonify([])
        
    try:
        dfs = []
        for i, m in enumerate(models):
            path = os.path.join(DATA_DIR, m, f'metrics_{size}.csv')
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, on_bad_lines='skip')
                    if 'Image' in df.columns and 'PSNR' in df.columns:
                        # Use copy() to avoid SettingWithCopy warnings/errors
                        new_df = df[['Image', 'PSNR']].rename(columns={'PSNR': f'PSNR_{i+1}'}).copy()
                        dfs.append(new_df)
                except Exception as e:
                    print(f"Error reading metrics for {m}: {e}")

        # Normalize all dataframes to use 'ID' for robust merging
        normalized_dfs = []
        for df_idx, df in enumerate(dfs):
            # Create 'ID' column by stripping extension
            df['ID'] = df['Image'].apply(lambda x: str(x).split('.')[0])
            df = df.drop_duplicates(subset=['ID'])
            normalized_dfs.append(df[['ID', f'PSNR_{df_idx+1}']])

        if not normalized_dfs:
            # Fallback to directory listing
            first_model_dir = os.path.join(DATA_DIR, models[0])
            images = []
            # Check standard subfolder or root
            search_paths = [os.path.join(first_model_dir, f'fid_real_{size}'), first_model_dir]
            for d in search_paths:
                if os.path.exists(d):
                    images.extend([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            unique_ids = sorted(list(set([f.split('.')[0] for f in images])))
            merged = pd.DataFrame({'ID': unique_ids})
            for i in range(len(models)): merged[f'PSNR_{i+1}'] = 0
        else:
            # Merge on 'ID'
            merged = normalized_dfs[0]
            for df in normalized_dfs[1:]:
                merged = pd.merge(merged, df, on='ID', how='outer')
            merged = merged.fillna(0)
    except Exception as e:
        print(f"Critical error in api_data processing: {e}")
        return jsonify({'error': str(e)}), 500
    
    # Load votes (optional now, but kept for compatibility)
    conn = sqlite3.connect('validation_results.db')
    c = conn.cursor()
    # Note: Voting logic was 1vs1, so we just use model1 and model2 for voting context if exists
    c.execute('SELECT image_id, winner, comment FROM votes WHERE size=? AND model1=? AND model2=?', (size, models[0], models[1] if len(models) > 1 else ''))
    votes_rows = c.fetchall()
    conn.close()
    
    votes_dict = {row[0]: {'winner': row[1], 'comment': row[2]} for row in votes_rows}
    
    # Get mask categories for filtering if using testing_mask_dataset
    indexed_masks = get_indexed_masks()
    valid_mask_paths = set(indexed_masks.get(size, []))
    # Extract just the filenames for easier matching
    valid_mask_names = {os.path.basename(p).split('.')[0] for p in valid_mask_paths}

    results = []
    for _, row in merged.iterrows():
        img_id = str(row['ID'])
        vote_data = votes_dict.get(img_id, {'winner': None, 'comment': ''})
        
        # Helper to find image path robustly
        def get_robust_path(model_name, subfolder_prefix, img_id_base):
            try:
                # 1. Try standard subfolder: fid_fake_SMALL/00000.png
                subfolder = f"{subfolder_prefix}_{size}"
                model_path = os.path.join(DATA_DIR, model_name)
                if not os.path.exists(model_path): return None
                
                exts = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
                
                target_sub = os.path.join(model_path, subfolder)
                if os.path.exists(target_sub):
                    for ext in exts:
                        if os.path.exists(os.path.join(target_sub, img_id_base + ext)):
                            return f'/images/{model_name}/{subfolder}/{img_id_base}{ext}'
                
                # 2. Try root folder second (flat structure)
                for ext in exts:
                    if os.path.exists(os.path.join(model_path, img_id_base + ext)):
                        return f'/images/{model_name}/{img_id_base}{ext}'
            except:
                pass
            return None

        gt_path = get_robust_path(models[0], 'fid_real', img_id) or ''

        item = {
            'id': img_id,
            'gt': gt_path,
            'winner': vote_data['winner'],
            'comment': vote_data['comment']
        }
        
        for i, m in enumerate(models):
            item[f'psnr_{i+1}'] = round(float(row.get(f'PSNR_{i+1}', 0)), 2)
            item[f'f{i+1}_fake'] = get_robust_path(m, 'fid_fake', img_id) or ''
            
        results.append(item)
    
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
    # Deterministic Indexing ONLY (No direct ID mapping)
    real_dir = os.path.join(DATA_DIR, model, f'fid_real_{size}')
    if not os.path.exists(real_dir):
        real_dir = os.path.join(DATA_DIR, model)
        
    if not os.path.exists(real_dir):
        return "Category directory not found", 404
        
    all_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_images.sort()
    
    image_filename = None
    for f in all_images:
        if f.startswith(image_name): 
            image_filename = f
            break
            
    if not image_filename:
        return f"Image {image_name} not found in {model}", 404
        
    img_index = all_images.index(image_filename)
    
    indexed_masks = get_indexed_masks()
    cat_masks = indexed_masks.get(size, [])
    
    if not cat_masks:
        return f"No masks found for category {size}", 404
        
    # Formula: (img_index * 2) % len(cat_masks)
    mask_path = cat_masks[(img_index * 2) % len(cat_masks)]
    
    try:
        with Image.open(mask_path) as mask_img:
            # Match current reference image size
            with Image.open(os.path.join(real_dir, image_filename)) as ref_img:
                w, h = ref_img.size
            mask_resized = mask_img.convert('L').resize((w, h), Image.NEAREST)
            buf = io.BytesIO()
            mask_resized.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
