import os
import glob
import pandas as pd
pd.options.compute.use_bottleneck = False
import json
import sqlite3
import zipfile
import tarfile
import shutil
import io
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template, request, send_from_directory, send_file

# Resolve template and static folders dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd_dir = os.getcwd()

if os.path.exists(os.path.join(script_dir, 'templates')):
    templates_path = os.path.join(script_dir, 'templates')
else:
    templates_path = os.path.join(cwd_dir, 'templates')

if os.path.exists(os.path.join(script_dir, 'static')):
    static_path = os.path.join(script_dir, 'static')
else:
    static_path = os.path.join(cwd_dir, 'static')

app = Flask(__name__, template_folder=templates_path, static_folder=static_path)

# Resolve DATA_DIR: check if current dir contains qualitative results or fall back
if os.path.exists(os.path.join(cwd_dir, 'TEMP_QUALITATIVE')):
    DATA_DIR = cwd_dir
else:
    script_data = os.path.abspath(os.path.join(script_dir, 'data'))
    cwd_data = os.path.abspath(os.path.join(cwd_dir, 'data'))
    if os.path.exists(script_data):
        DATA_DIR = script_data
    elif os.path.exists(cwd_data):
        DATA_DIR = cwd_data
    else:
        DATA_DIR = cwd_dir

print(f"[*] App initialization: using DATA_DIR = {DATA_DIR}")

# Resolve Database path
DB_PATH = os.path.join(script_dir, 'validation_results.db')
if not os.access(os.path.dirname(DB_PATH) or '.', os.W_OK):
    DB_PATH = os.path.join(DATA_DIR, 'validation_results.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if old table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='votes'")
    if c.fetchone():
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

# --- Auto-Extraction logic ---
def check_and_extract_archives():
    # 1. Unzip testing_mask_dataset.zip if directory doesn't exist
    mask_dir = os.path.join(DATA_DIR, 'testing_mask_dataset')
    if not os.path.exists(mask_dir) or not os.listdir(mask_dir):
        zip_candidates = [
            os.path.join(DATA_DIR, 'testing_mask_dataset.zip'),
            os.path.join(cwd_dir, 'testing_mask_dataset.zip'),
            os.path.join(script_dir, 'testing_mask_dataset.zip'),
            os.path.join(script_dir, 'datasets', 'testing_mask_dataset.zip')
        ]
        for zip_path in zip_candidates:
            if os.path.exists(zip_path):
                print(f"[+] Auto-extracting masks: {zip_path} -> {DATA_DIR}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(DATA_DIR)
                    print("[+] Successfully extracted mask dataset.")
                    break
                except Exception as e:
                    print(f"[-] Error extracting mask dataset: {e}")
                    
    # 2. Extract Ground Truth archives if directory doesn't exist
    gt_dir = os.path.join(DATA_DIR, 'test_256')
    celeba_gt = os.path.join(DATA_DIR, 'celeba_hq_256_test')
    if (not os.path.exists(gt_dir) or not os.listdir(gt_dir)) and (not os.path.exists(celeba_gt) or not os.listdir(celeba_gt)):
        archive_candidates = [
            (os.path.join(DATA_DIR, 'test_256.tar'), 'tar'),
            (os.path.join(cwd_dir, 'test_256.tar'), 'tar'),
            (os.path.join(script_dir, 'test_256.tar'), 'tar'),
            (os.path.join(DATA_DIR, 'celeba_hq_256_test.zip'), 'zip'),
            (os.path.join(cwd_dir, 'celeba_hq_256_test.zip'), 'zip'),
            (os.path.join(script_dir, 'celeba_hq_256_test.zip'), 'zip')
        ]
        for arch_path, fmt in archive_candidates:
            if os.path.exists(arch_path):
                print(f"[+] Auto-extracting Ground Truth: {arch_path} -> {DATA_DIR}...")
                try:
                    if fmt == 'tar':
                        with tarfile.open(arch_path, 'r') as tar_ref:
                            tar_ref.extractall(DATA_DIR)
                    elif fmt == 'zip':
                        with zipfile.ZipFile(arch_path, 'r') as zip_ref:
                            zip_ref.extractall(DATA_DIR)
                    print("[+] Successfully extracted Ground Truth dataset.")
                    break
                except Exception as e:
                    print(f"[-] Error extracting GT dataset: {e}")

def ensure_frontend_assets():
    """Ensure templates and static files exist in the directory where server is run"""
    for folder, path_var in [('templates', templates_path), ('static', static_path)]:
        os.makedirs(path_var, exist_ok=True)
        src_candidates = [
            os.path.join(script_dir, folder),
            os.path.join(script_dir, 'validation_dashboard', folder)
        ]
        for src_folder in src_candidates:
            if src_folder != path_var and os.path.exists(src_folder):
                print(f"[+] Copying {folder} folder assets from {src_folder} to {path_var}...")
                for item in os.listdir(src_folder):
                    src_item = os.path.join(src_folder, item)
                    dst_item = os.path.join(path_var, item)
                    if os.path.isdir(src_item):
                        shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_item, dst_item)
                break

# --- Helper logic for recursive directories and qualitative files ---
indexed_masks_cache = None

def get_indexed_masks():
    global indexed_masks_cache
    if indexed_masks_cache:
        return indexed_masks_cache
        
    mask_dir = os.path.join(DATA_DIR, 'testing_mask_dataset')
    if not os.path.exists(mask_dir):
        return {'SMALL': [], 'MEDIUM': [], 'LARGE': []}
        
    try:
        mask_files = [os.path.join(mask_dir, entry.name) for entry in os.scandir(mask_dir)
                      if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception:
        mask_files = []
        
    mask_files.sort()
    
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

def find_model_folders(base_dir):
    model_folders = []
    excluded = [
        'testing_mask_dataset', 'masks', 'iregularmask', 'venv', 
        'temp_', 'static', 'templates', 'test_256', 'datasets',
        '.git', '.github', 'PlacesDateset', 'PlacesTraining'
    ]
    
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded]
        has_metrics = any(f.startswith('metrics_') and f.endswith('.csv') for f in files)
        if has_metrics:
            rel_path = os.path.relpath(root, base_dir)
            if rel_path == '.':
                continue
            rel_path = rel_path.replace('\\', '/')
            model_folders.append(rel_path)
            
    model_folders.sort()
    return model_folders

def get_size_folder(model_path, size):
    d1 = os.path.join(model_path, size)
    if os.path.isdir(d1):
        return d1
    d2 = os.path.join(model_path, f"fid_fake_{size}")
    if os.path.isdir(d2):
        return d2
    d3 = os.path.join(model_path, size.lower())
    if os.path.isdir(d3):
        return d3
    d4 = os.path.join(model_path, f"fid_fake_{size.lower()}")
    if os.path.isdir(d4):
        return d4
    return model_path

def parse_filename(filename):
    """
    Parses [image_id]_[mask_id]_[psnr].png format, e.g. Places365_test_00162032_09972_27.40.png
    """
    base, ext = os.path.splitext(filename)
    parts = base.split('_')
    
    if len(parts) >= 3:
        try:
            float(parts[-1])
            is_psnr = True
        except ValueError:
            is_psnr = False
            
        if is_psnr:
            mask_id = parts[-2]
            image_id = '_'.join(parts[:-2])
            return image_id, mask_id, parts[-1]
            
    return base, None, None

# --- Path Caching Mechanism ---
CACHE_FILE_PATH = os.path.join(DATA_DIR, 'paths_cache.json')
paths_cache = None

def load_or_build_cache(rebuild=False):
    global paths_cache
    if paths_cache and not rebuild:
        return paths_cache
        
    if os.path.exists(CACHE_FILE_PATH) and not rebuild:
        print("[*] Loading file paths from paths_cache.json...")
        try:
            with open(CACHE_FILE_PATH, 'r') as f:
                paths_cache = json.load(f)
            return paths_cache
        except Exception as e:
            print(f"[-] Error reading paths_cache.json, rebuilding: {e}")
            
    print("[*] Building file paths cache (paths_cache.json)...")
    
    # 1. Model folders
    model_folders = find_model_folders(DATA_DIR)
    
    # 2. Ground Truth images cache
    gt_images = {}
    possible_gt_dirs = [
        os.path.join(DATA_DIR, 'test_256'),
        os.path.join(DATA_DIR, 'datasets', 'places365', 'test_256'),
        os.path.join(DATA_DIR, 'PlacesDateset'),
    ]
    exts = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    
    for gt_dir in possible_gt_dirs:
        if os.path.exists(gt_dir):
            for root, dirs, files in os.walk(gt_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for f in files:
                    f_base, f_ext = os.path.splitext(f)
                    if f_ext in exts:
                        if f_base not in gt_images:
                            full_p = os.path.join(root, f)
                            rel_p = os.path.relpath(full_p, DATA_DIR).replace('\\', '/')
                            gt_images[f_base] = rel_p
                            
    # Fallback to model's fid_real subfolders for GT
    for mf in model_folders:
        for size in ['LARGE', 'MEDIUM', 'SMALL']:
            real_dir = os.path.join(DATA_DIR, mf, f'fid_real_{size}')
            if os.path.exists(real_dir):
                try:
                    for entry in os.scandir(real_dir):
                        if entry.is_file():
                            f_base, f_ext = os.path.splitext(entry.name)
                            if f_ext in exts and f_base not in gt_images:
                                rel_p = os.path.relpath(entry.path, DATA_DIR).replace('\\', '/')
                                gt_images[f_base] = rel_p
                except Exception:
                    pass

    # 3. Mask images cache
    mask_images = {}
    mask_dir = os.path.join(DATA_DIR, 'testing_mask_dataset')
    if os.path.exists(mask_dir):
        try:
            for entry in os.scandir(mask_dir):
                if entry.is_file():
                    f_base, f_ext = os.path.splitext(entry.name)
                    if f_ext in exts:
                        rel_p = os.path.relpath(entry.path, DATA_DIR).replace('\\', '/')
                        mask_images[f_base] = rel_p
        except Exception:
            pass

    # 4. Model outputs cache
    model_images = {}
    for mf in model_folders:
        model_images[mf] = {}
        for size in ['LARGE', 'MEDIUM', 'SMALL']:
            model_images[mf][size] = {}
            model_dir = os.path.join(DATA_DIR, mf)
            size_folder = get_size_folder(model_dir, size)
            if os.path.exists(size_folder):
                try:
                    size_folder_rel = os.path.relpath(size_folder, DATA_DIR).replace('\\', '/')
                    for entry in os.scandir(size_folder):
                        if entry.is_file():
                            f_base, f_ext = os.path.splitext(entry.name)
                            if f_ext in exts:
                                # Map exact name
                                model_images[mf][size][f_base] = f"{size_folder_rel}/{entry.name}"
                                # Parse qualitative format
                                parsed_id, _, _ = parse_filename(entry.name)
                                if parsed_id:
                                    model_images[mf][size][parsed_id] = f"{size_folder_rel}/{entry.name}"
                                    # Numeric fallback
                                    digits = ''.join(c for c in parsed_id if c.isdigit())
                                    if digits:
                                        model_images[mf][size][digits] = f"{size_folder_rel}/{entry.name}"
                except Exception as e:
                    print(f"[-] Error mapping size folder {size} for {mf}: {e}")

    paths_cache = {
        "model_folders": model_folders,
        "gt_images": gt_images,
        "mask_images": mask_images,
        "model_images": model_images
    }

    try:
        with open(CACHE_FILE_PATH, 'w') as f:
            json.dump(paths_cache, f, indent=2)
        print(f"[+] Saved paths cache file to {CACHE_FILE_PATH}")
    except Exception as e:
        print(f"[-] Failed to write cache to file: {e}")
        
    return paths_cache

# Call startup init functions
check_and_extract_archives()
ensure_frontend_assets()
init_db()
load_or_build_cache()

# --- API Endpoints ---

@app.route('/api/folders')
def api_folders():
    rebuild = request.args.get('rebuild', 'false').lower() == 'true'
    cache = load_or_build_cache(rebuild=rebuild)
    return jsonify(cache["model_folders"])

@app.route('/api/data')
def api_data():
    size = request.args.get('size', 'SMALL')
    models = [request.args.get(f'model{i}') for i in range(1, 6)]
    models = [m for m in models if m]
    
    if not models:
        return jsonify([])
        
    rebuild = request.args.get('rebuild', 'false').lower() == 'true'
    cache = load_or_build_cache(rebuild=rebuild)
    
    gt_cache = cache["gt_images"]
    model_images = cache["model_images"]
    
    try:
        # Process metrics CSVs
        dfs = []
        for i, m in enumerate(models):
            path = os.path.join(DATA_DIR, m, f'metrics_{size}.csv')
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, on_bad_lines='skip')
                    df.columns = [c.strip() for c in df.columns]
                    if 'Image' in df.columns and 'PSNR' in df.columns:
                        df = df[df['Image'].notna()]
                        df = df[df['Image'].str.strip() != '']
                        df = df[df['Image'].str.strip() != 'AVERAGE']
                        
                        new_df = df[['Image', 'PSNR']].rename(columns={'PSNR': f'PSNR_{i+1}'}).copy()
                        dfs.append(new_df)
                except Exception as e:
                    print(f"Error reading metrics for {m}: {e}")

        normalized_dfs = []
        for df in dfs:
            df['ID'] = df['Image'].apply(lambda x: str(x).split('.')[0].strip())
            df = df.drop_duplicates(subset=['ID'])
            psnr_col = next((c for c in df.columns if c.startswith('PSNR_')), None)
            if psnr_col:
                normalized_dfs.append(df[['ID', psnr_col]])

        if not normalized_dfs:
            # Listing images from the preloaded mapping of the first model
            first_model = models[0]
            mapping = model_images.get(first_model, {}).get(size, {})
            unique_ids = sorted(list(set(mapping.keys())))
            merged = pd.DataFrame({'ID': unique_ids})
            for i in range(len(models)):
                merged[f'PSNR_{i+1}'] = 0.0
        else:
            merged = normalized_dfs[0]
            for df in normalized_dfs[1:]:
                merged = pd.merge(merged, df, on='ID', how='outer')
            merged = merged.fillna(0.0)
    except Exception as e:
        print(f"Critical error in api_data processing: {e}")
        return jsonify({'error': str(e)}), 500
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT image_id, winner, comment FROM votes WHERE size=? AND model1=? AND model2=?', 
              (size, models[0], models[1] if len(models) > 1 else ''))
    votes_rows = c.fetchall()
    conn.close()
    
    votes_dict = {row[0]: {'winner': row[1], 'comment': row[2]} for row in votes_rows}
    
    results = []
    for _, row in merged.iterrows():
        img_id = str(row['ID'])
        vote_data = votes_dict.get(img_id, {'winner': None, 'comment': ''})
        
        gt_rel = gt_cache.get(img_id)
        if gt_rel:
            gt_path = f'/images/{gt_rel}'
        else:
            gt_path = ''
            
        item = {
            'id': img_id,
            'gt': gt_path,
            'winner': vote_data['winner'],
            'comment': vote_data['comment']
        }
        
        for i, m in enumerate(models):
            mapping = model_images.get(m, {}).get(size, {})
            rel_img_path = mapping.get(img_id)
            if not rel_img_path:
                digits = ''.join(c for c in img_id if c.isdigit())
                rel_img_path = mapping.get(digits)
                
            if rel_img_path:
                item[f'f{i+1}_fake'] = f'/images/{rel_img_path}'
            else:
                item[f'f{i+1}_fake'] = ''
                
            val = row.get(f'PSNR_{i+1}', 0.0)
            if (val == 0.0 or pd.isna(val) or val == '0') and rel_img_path:
                img_file = os.path.basename(rel_img_path)
                _, _, parsed_psnr = parse_filename(img_file)
                if parsed_psnr:
                    val = float(parsed_psnr)
            item[f'psnr_{i+1}'] = round(float(val), 2) if (not pd.isna(val) and val) else 0.0
            
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
        
    conn = sqlite3.connect(DB_PATH)
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
        
    conn = sqlite3.connect(DB_PATH)
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
    
    downloaded_chunks = len(os.listdir(temp_dir))
    if downloaded_chunks == total_chunks:
        final_zip_path = os.path.join(DATA_DIR, filename)
        try:
            with open(final_zip_path, 'wb') as outfile:
                for i in range(total_chunks):
                    cp = os.path.join(temp_dir, str(i))
                    with open(cp, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
            
            shutil.rmtree(temp_dir)
            
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
        
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            os.remove(filepath)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(DATA_DIR, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/mask_only/<path:model>/<size>/<image_name>')
def api_mask_only(model, size, image_name):
    cache = load_or_build_cache()
    
    mapping = cache["model_images"].get(model, {}).get(size, {})
    rel_img_path = mapping.get(image_name)
    if not rel_img_path:
        digits = ''.join(c for c in image_name if c.isdigit())
        rel_img_path = mapping.get(digits)
        
    mask_id = None
    if rel_img_path:
        img_file = os.path.basename(rel_img_path)
        _, parsed_mask_id, _ = parse_filename(img_file)
        if parsed_mask_id:
            mask_id = parsed_mask_id
            
    mask_rel_path = cache["mask_images"].get(mask_id) if mask_id else None
    
    # Fallback to deterministic index-based matching
    if not mask_rel_path:
        indexed_masks = get_indexed_masks()
        cat_masks = indexed_masks.get(size, [])
        if cat_masks:
            all_images = sorted(list(set(mapping.values())))
            
            image_filename = None
            for f_rel in all_images:
                f = os.path.basename(f_rel)
                parsed_id, _, _ = parse_filename(f)
                if parsed_id == image_name or f.startswith(image_name):
                    image_filename = f_rel
                    break
                    
            if image_filename and image_filename in all_images:
                img_index = all_images.index(image_filename)
                mask_full_path = cat_masks[(img_index * 2) % len(cat_masks)]
                mask_rel_path = os.path.relpath(mask_full_path, DATA_DIR).replace('\\', '/')
                
    if not mask_rel_path:
        return f"Mask not found for image {image_name} (parsed mask_id: {mask_id})", 404
        
    try:
        mask_path = os.path.join(DATA_DIR, mask_rel_path)
        with Image.open(mask_path) as mask_img:
            ref_path = None
            if rel_img_path:
                ref_path = os.path.join(DATA_DIR, rel_img_path)
            else:
                gt_rel = cache["gt_images"].get(image_name)
                if gt_rel:
                    ref_path = os.path.join(DATA_DIR, gt_rel)
                    
            if ref_path and os.path.exists(ref_path):
                with Image.open(ref_path) as ref_img:
                    w, h = ref_img.size
            else:
                w, h = 256, 256
                
            mask_resized = mask_img.convert('L').resize((w, h), Image.NEAREST)
            buf = io.BytesIO()
            mask_resized.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    print("[*] Qualitative Dashboard running on Localhost:")
    print("[*] Open in browser: http://127.0.0.1:5003 or http://localhost:5003")
    app.run(host='0.0.0.0', port=5003, debug=True)
