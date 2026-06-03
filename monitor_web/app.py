import os
import glob
import json
import time
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation_results_test', '5_image_grid'))

def get_stats():
    files = glob.glob(os.path.join(RESULTS_DIR, '**/*.png'), recursive=True)
    # Sorting by creation time to get chronological view if needed, or by filename
    files.sort(key=os.path.getmtime, reverse=True)
    
    stats = {
        'SMALL': {'count': 0, 'psnr_sum': 0, 'data': []},
        'MEDIUM': {'count': 0, 'psnr_sum': 0, 'data': []},
        'LARGE': {'count': 0, 'psnr_sum': 0, 'data': []},
        'OTHER': {'count': 0, 'psnr_sum': 0, 'data': []},
        'total_count': len(files),
        'latest_images': []
    }
    
    psnr_series = [] # For the graph

    # We process in chronological order for the graph (oldest to newest)
    all_data = []
    
    for f in sorted(files, key=os.path.getmtime):
        rel_path = os.path.relpath(f, RESULTS_DIR)
        name = os.path.basename(f)
        parts = name.split('_')
        if len(parts) >= 3:
            category = parts[0]
            try:
                psnr = float(parts[-1].replace('.png', ''))
            except:
                continue
            
            if category not in stats:
                category = 'OTHER'
                
            stats[category]['count'] += 1
            stats[category]['psnr_sum'] += psnr
            
            entry = {
                'name': rel_path, # Use relative path for image serving
                'display_name': name,
                'psnr': psnr,
                'time': os.path.getmtime(f)
            }
            all_data.append(entry)
            psnr_series.append(psnr)

    # Calculate average time per image by grouping file writes into batches
    avg_time = 0.0
    sorted_times = sorted([x['time'] for x in all_data])
    if len(sorted_times) > 1:
        batch_completions = []
        last_t = sorted_times[0]
        curr_count = 1
        
        for t in sorted_times[1:]:
            if t - last_t > 5.0: # New batch detected (gap larger than 5s)
                batch_completions.append((last_t, curr_count))
                curr_count = 1
            else:
                curr_count += 1
            last_t = t
        batch_completions.append((last_t, curr_count))
        
        batch_deltas = []
        # Calculate time delta between consecutive batches divided by batch size
        for i in range(1, len(batch_completions)):
            delta = batch_completions[i][0] - batch_completions[i-1][0]
            # Ignore gaps where delta is too large (paused or restarted)
            if delta < 120:
                batch_deltas.append(delta / batch_completions[i][1])
                
        avg_time = round(sum(batch_deltas) / len(batch_deltas), 3) if batch_deltas else 0.0

    # Calculate averages
    for cat in ['SMALL', 'MEDIUM', 'LARGE', 'OTHER']:
        if stats[cat]['count'] > 0:
            stats[cat]['avg_psnr'] = round(stats[cat]['psnr_sum'] / stats[cat]['count'], 2)
        else:
            stats[cat]['avg_psnr'] = 0

    # Calculate ETA
    TOTAL_TARGET = 316000
    remaining = max(0, TOTAL_TARGET - len(all_data))
    eta_seconds = remaining * avg_time
    
    # Format ETA as HH:MM:SS
    hours = int(eta_seconds // 3600)
    minutes = int((eta_seconds % 3600) // 60)
    seconds = int(eta_seconds % 60)
    eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Latest 20 images for the UI feed
    stats['latest_images'] = sorted(all_data, key=lambda x: x['time'], reverse=True)[:20]
    stats['psnr_series'] = psnr_series[-100:] # Last 100 points for the live graph
    stats['avg_gen_time'] = avg_time
    stats['eta'] = eta_str
    stats['remaining'] = remaining

    return stats

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
