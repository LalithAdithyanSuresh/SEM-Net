const API_BASE = '/api';

// UI Elements
const btnRun = document.getElementById('btn-run');
const btnStop = document.getElementById('btn-stop');
const btnPull = document.getElementById('btn-pull');
const cmdBadge = document.getElementById('current-command-badge');
const modelBadge = document.getElementById('current-model-badge');
const modelSelector = document.getElementById('model-config');
const connectionStatus = document.getElementById('connection-status');
const terminalOutput = document.getElementById('terminal-output');
const imageGallery = document.getElementById('image-gallery');
const imageCount = document.getElementById('image-count');

// Chart Setup
let metricsChart;
function initChart() {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'L1 Loss', data: [], borderColor: '#3b82f6', tension: 0.4, borderWidth: 2 },
                { label: 'Perceptual', data: [], borderColor: '#10b981', tension: 0.4, borderWidth: 2 },
                { label: 'Adversarial', data: [], borderColor: '#f59e0b', tension: 0.4, borderWidth: 2 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', maxTicksLimit: 10 } }
            },
            animation: { duration: 0 }
        }
    });
}

// Format Terminal Output
function formatLogLine(line) {
    if (line.toLowerCase().includes('error')) return `<span class="log-error">${line}</span>`;
    if (line.toLowerCase().includes('warning')) return `<span class="log-warn">${line}</span>`;
    if (line.includes('Epoch') || line.includes('INFO')) return `<span class="log-info">${line}</span>`;
    return line;
}

// Fetch State
async function fetchState() {
    try {
        const res = await fetch(`${API_BASE}/command`);
        const data = await res.json();
        
        connectionStatus.className = 'pulse-dot online';
        
        // Update badges
        cmdBadge.textContent = data.command.toUpperCase();
        cmdBadge.className = `badge badge-${data.command === 'run' ? 'active' : (data.command === 'stop' ? 'stopped' : 'pull')}`;
        
        modelBadge.textContent = data.model;
        if(document.activeElement !== modelSelector) {
            modelSelector.value = data.model;
        }

    } catch (e) {
        connectionStatus.className = 'pulse-dot'; // Offline
        console.error('Failed to fetch state');
    }
}

// Fetch Available Models
let knownModels = new Set();
async function fetchModels() {
    try {
        const res = await fetch(`${API_BASE}/available_models`);
        const data = await res.json();
        const models = data.models || [];
        
        // If models changed, update dropdown
        if (models.length > 0 && models.some(m => !knownModels.has(m) || knownModels.size !== models.length)) {
            const currentValue = modelSelector.value;
            modelSelector.innerHTML = ''; // clear
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                modelSelector.appendChild(opt);
                knownModels.add(m);
            });
            // restore selected
            if(models.includes(currentValue)) {
                modelSelector.value = currentValue;
            }
        }
    } catch(e) {}
}

// Fetch Metrics
async function fetchMetrics() {
    try {
        const res = await fetch(`${API_BASE}/metrics`);
        const data = await res.json();
        
        if (data.length > 0 && metricsChart) {
            const labels = data.map(d => `Ep ${d.epoch}`);
            const l1 = data.map(d => d.val_gen_l1);
            const pl = data.map(d => d.val_gen_pl);
            const adv = data.map(d => d.val_gen_adv);
            
            // Only update if data changed
            if (metricsChart.data.labels.length !== labels.length) {
                metricsChart.data.labels = labels;
                metricsChart.data.datasets[0].data = l1;
                metricsChart.data.datasets[1].data = pl;
                metricsChart.data.datasets[2].data = adv;
                metricsChart.update();
            }
        }
    } catch (e) {}
}

let lastLogHash = 0;
// Fetch Logs
async function fetchLogs() {
    try {
        const res = await fetch(`${API_BASE}/logs`);
        const lines = await res.json();
        
        if (lines.length > 0 && lines.length !== lastLogHash) {
            lastLogHash = lines.length;
            terminalOutput.innerHTML = lines.map(l => `<div class="terminal-line">${formatLogLine(l)}</div>`).join('');
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
        }
    } catch (e) {}
}

let loadedImages = new Set();
// Fetch Images
async function fetchImages() {
    try {
        const res = await fetch(`${API_BASE}/images`);
        const images = await res.json();
        
        imageCount.textContent = images.length;
        
        if(images.length > 0) {
            if(loadedImages.size === 0) imageGallery.innerHTML = ''; // clear empty state
            
            images.forEach(img => {
                if(!loadedImages.has(img)) {
                    loadedImages.add(img);
                    const div = document.createElement('div');
                    div.className = 'gallery-item';
                    div.innerHTML = `
                        <img src="/images/${img}" alt="${img}" loading="lazy">
                        <div class="overlay">${img}</div>
                    `;
                    imageGallery.insertBefore(div, imageGallery.firstChild);
                }
            });
        }
    } catch (e) {}
}

// Actions
async function sendCommand(cmd) {
    const model = modelSelector.value;
    await fetch(`${API_BASE}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd, model: model })
    });
    fetchState();
}

btnRun.addEventListener('click', () => sendCommand('run'));
btnStop.addEventListener('click', () => sendCommand('stop'));
btnPull.addEventListener('click', () => sendCommand('restart_pull'));

// Bootstrap
initChart();

// Polling Loops
setInterval(fetchState, 1000);
setInterval(fetchModels, 2000);
setInterval(fetchLogs, 1000);
setInterval(fetchMetrics, 3000);
setInterval(fetchImages, 5000);

