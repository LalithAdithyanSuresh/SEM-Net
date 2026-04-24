import os
import glob
import json
import time
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

RESULTS_DIR = os.path.abspath('../evaluation_results_standard_uniform/5_image_grid')

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

    # Calculate average time per image (based on last 30 files for accuracy)
    time_deltas = []
    sorted_by_time = sorted(all_data, key=lambda x: x['time'])
    if len(sorted_by_time) > 1:
        # Look at last 30 intervals
        recent_data = sorted_by_time[-31:]
        for i in range(1, len(recent_data)):
            delta = recent_data[i]['time'] - recent_data[i-1]['time']
            # Filter out deltas larger than 2 minutes (assuming gaps in evaluation)
            if delta < 120:
                time_deltas.append(delta)
    
    avg_time = round(sum(time_deltas) / len(time_deltas), 2) if time_deltas else 0

    # Calculate averages
    for cat in ['SMALL', 'MEDIUM', 'LARGE', 'OTHER']:
        if stats[cat]['count'] > 0:
            stats[cat]['avg_psnr'] = round(stats[cat]['psnr_sum'] / stats[cat]['count'], 2)
        else:
            stats[cat]['avg_psnr'] = 0

    # Calculate ETA
    TOTAL_TARGET = 6000
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
