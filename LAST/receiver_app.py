import time
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Global storage for the latest test run stats
latest_stats = {
    "SMALL": {"completions": 0, "total": 2000, "psnr": 0.0, "done": False},
    "MEDIUM": {"completions": 0, "total": 2000, "psnr": 0.0, "done": False},
    "LARGE": {"completions": 0, "total": 2000, "psnr": 0.0, "done": False},
    "last_updated": 0
}

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEM-Net Evaluation Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
            --accent-orange: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
        }

        header {
            padding: 2rem;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
        }

        h1 {
            font-weight: 800;
            font-size: 1.8rem;
            letter-spacing: -0.05em;
            background: linear-gradient(to right, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            background-color: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(59, 130, 246, 0.3);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .pulse {
            width: 8px;
            height: 8px;
            background-color: var(--accent-blue);
            border-radius: 50%;
            box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
            animation: pulse-animation 1.5s infinite;
        }

        @keyframes pulse-animation {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 8px rgba(59, 130, 246, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
            }
        }

        main {
            flex-grow: 1;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
        }

        .card {
            background: var(--card-bg);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            transition: transform 0.2s, border-color 0.2s;
            position: relative;
            overflow: hidden;
        }

        .card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.15);
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }

        .card-small::before { background-color: var(--accent-green); }
        .card-medium::before { background-color: var(--accent-blue); }
        .card-large::before { background-color: var(--accent-purple); }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }

        .cat-title {
            font-size: 1.4rem;
            font-weight: 600;
        }

        .cat-done-badge {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-box {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 0.75rem;
            border: 1px solid var(--border-color);
        }

        .metric-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-size: 1.25rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .progress-container {
            margin-top: 1rem;
        }

        .progress-label-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .progress-bar-bg {
            background: rgba(255, 255, 255, 0.05);
            height: 10px;
            border-radius: 9999px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.4s ease-out;
            width: 0%;
        }

        .card-small .progress-bar-fill { background: var(--accent-green); }
        .card-medium .progress-bar-fill { background: var(--accent-blue); }
        .card-large .progress-bar-fill { background: var(--accent-purple); }

        footer {
            padding: 2rem;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color);
            margin-top: 2rem;
            max-width: 1200px;
            width: 100%;
            margin-left: auto;
            margin-right: auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        #last-updated-text {
            font-family: 'JetBrains Mono', monospace;
        }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>SEM-Net Evaluation</h1>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.25rem;">Live tracking from test server</p>
        </div>
        <div class="status-badge">
            <div class="pulse"></div>
            <span id="conn-status">Active</span>
        </div>
    </header>

    <main>
        <div class="grid-container">
            <!-- SMALL CARD -->
            <div class="card card-small">
                <div class="card-header">
                    <span class="cat-title">Small Masks</span>
                    <span id="badge-SMALL" class="cat-done-badge" style="display:none;">Done</span>
                </div>
                <div class="metrics">
                    <div class="metric-box">
                        <div class="metric-label">Completions</div>
                        <div id="comp-SMALL" class="metric-value">0 / 2000</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Avg PSNR</div>
                        <div id="psnr-SMALL" class="metric-value">0.00</div>
                    </div>
                </div>
                <div class="progress-container">
                    <div class="progress-label-row">
                        <span>Progress</span>
                        <span id="percent-SMALL">0%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div id="fill-SMALL" class="progress-bar-fill"></div>
                    </div>
                </div>
            </div>

            <!-- MEDIUM CARD -->
            <div class="card card-medium">
                <div class="card-header">
                    <span class="cat-title">Medium Masks</span>
                    <span id="badge-MEDIUM" class="cat-done-badge" style="display:none;">Done</span>
                </div>
                <div class="metrics">
                    <div class="metric-box">
                        <div class="metric-label">Completions</div>
                        <div id="comp-MEDIUM" class="metric-value">0 / 2000</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Avg PSNR</div>
                        <div id="psnr-MEDIUM" class="metric-value">0.00</div>
                    </div>
                </div>
                <div class="progress-container">
                    <div class="progress-label-row">
                        <span>Progress</span>
                        <span id="percent-MEDIUM">0%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div id="fill-MEDIUM" class="progress-bar-fill"></div>
                    </div>
                </div>
            </div>

            <!-- LARGE CARD -->
            <div class="card card-large">
                <div class="card-header">
                    <span class="cat-title">Large Masks</span>
                    <span id="badge-LARGE" class="cat-done-badge" style="display:none;">Done</span>
                </div>
                <div class="metrics">
                    <div class="metric-box">
                        <div class="metric-label">Completions</div>
                        <div id="comp-LARGE" class="metric-value">0 / 2000</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Avg PSNR</div>
                        <div id="psnr-LARGE" class="metric-value">0.00</div>
                    </div>
                </div>
                <div class="progress-container">
                    <div class="progress-label-row">
                        <span>Progress</span>
                        <span id="percent-LARGE">0%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div id="fill-LARGE" class="progress-bar-fill"></div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer>
        <div>SEM-Net © 2026</div>
        <div id="last-updated-text">Waiting for data...</div>
    </footer>

    <script>
        async function fetchStats() {
            try {
                const res = await fetch(window.location.pathname + '/stats');
                if (!res.ok) throw new Error("Offline");
                const data = await res.json();
                
                document.getElementById('conn-status').innerText = "Active";
                document.getElementById('conn-status').parentElement.style.color = "var(--accent-blue)";
                
                ['SMALL', 'MEDIUM', 'LARGE'].forEach(cat => {
                    const c = data[cat];
                    const percentage = c.total > 0 ? Math.round((c.completions / c.total) * 100) : 0;
                    
                    document.getElementById(`comp-${cat}`).innerText = `${c.completions} / ${c.total}`;
                    document.getElementById(`psnr-${cat}`).innerText = c.psnr > 0 ? c.psnr.toFixed(2) : "0.00";
                    document.getElementById(`percent-${cat}`).innerText = `${percentage}%`;
                    document.getElementById(`fill-${cat}`).style.width = `${percentage}%`;
                    
                    if (c.done) {
                        document.getElementById(`badge-${cat}`).style.display = "inline-block";
                    } else {
                        document.getElementById(`badge-${cat}`).style.display = "none";
                    }
                });

                if (data.last_updated > 0) {
                    const timeDiff = Math.round(Date.now() / 1000 - data.last_updated);
                    document.getElementById('last-updated-text').innerText = `Last updated: ${timeDiff}s ago`;
                } else {
                    document.getElementById('last-updated-text').innerText = "No updates received yet";
                }

            } catch (err) {
                console.error(err);
                document.getElementById('conn-status').innerText = "Offline";
                document.getElementById('conn-status').parentElement.style.color = "var(--accent-orange)";
            }
        }

        // Poll every 2 seconds for real-time responsiveness
        setInterval(fetchStats, 2000);
        fetchStats();
    </script>
</body>
</html>
"""

@app.route('/camino-places', methods=['GET'])
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/camino-places/stats', methods=['GET'])
def stats():
    return jsonify(latest_stats)

@app.route('/camino-places', methods=['POST'])
def update():
    global latest_stats
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload"}), 400
        
    for cat in ["SMALL", "MEDIUM", "LARGE"]:
        if cat in data:
            latest_stats[cat] = data[cat]
            
    latest_stats["last_updated"] = int(time.time())
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
