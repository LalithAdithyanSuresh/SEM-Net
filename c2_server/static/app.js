const API_BASE = '/api';

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════
let selectedRun = '';
let allMetricsCache = [];   // full structured metrics for current view
let lastLogCount = 0;       // tracks total lines received for delta fetching
let currentTab = 'losses';  // 'losses' | 'quality'

const runSelector = document.getElementById('run-selector');

// ═══════════════════════════════════════════════════════════════
// Viper / Snake Mamba Colorscheme 🐍
// ═══════════════════════════════════════════════════════════════
const LOSS_DATASETS = [
    { key: 'gen_loss',        label: 'Gen Loss (total)', color: '#00ff88', axis: 'y' }, // neon green
    { key: 'dis_loss',        label: 'Dis Loss',         color: '#ff6b00', axis: 'y' }, // vivid orange
    { key: 'l1_loss',         label: 'L1 Loss',          color: '#00cfff', axis: 'y' }, // electric cyan
    { key: 'perceptual_loss', label: 'Perceptual',       color: '#ff2d78', axis: 'y' }, // hot pink
    { key: 'style_loss',      label: 'Style',            color: '#c97dff', axis: 'y' }, // vivid violet
    { key: 'sym_loss',        label: 'Symmetry',         color: '#ffe600', axis: 'y' }, // neon yellow
];

const QUALITY_DATASETS = [
    { key: 'psnr', label: 'PSNR (dB)', color: '#00ff88', axis: 'y'  }, // neon green
    { key: 'mae',  label: 'MAE',       color: '#ff6b00', axis: 'y2' }, // vivid orange
];

function smooth(arr, w) {
    if (w <= 1) return arr;
    const out = [];
    for (let i = 0; i < arr.length; i++) {
        const start = Math.max(0, i - w + 1);
        const slice = arr.slice(start, i + 1);
        out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════
// Plotly Chart
// ═══════════════════════════════════════════════════════════════
function renderChart(data) {
    if (data.length === 0) return;
    const w = Math.max(1, parseInt(document.getElementById('smooth-window').value) || 10);
    const defs = currentTab === 'losses' ? LOSS_DATASETS : QUALITY_DATASETS;

    const iterations = data.map(d => d.iteration);
    
    const plotData = defs.map(d => {
        const raw = data.map(pt => pt[d.key] != null ? pt[d.key] : null);
        const smoothed = smooth(raw.map(v => v ?? 0), w);
        return {
            x: iterations,
            y: smoothed,
            name: d.label,
            type: 'scatter',
            mode: 'lines',
            line: { color: d.color, width: 2.5 },
            fill: 'tozeroy',
            fillcolor: d.color + '22', // ~13% opacity fill
            yaxis: d.axis
        };
    });

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Outfit', color: '#94a3b8' },
        margin: { t: 10, r: 40, l: 40, b: 40 },
        hovermode: 'x unified',
        dragmode: 'zoom',
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.1)'
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.1)'
        },
        legend: { orientation: 'h', y: 1.1 },
    };

    if (currentTab === 'quality') {
        layout.yaxis2 = {
            overlaying: 'y',
            side: 'right',
            showgrid: false,
            zeroline: false
        };
    }

    Plotly.react('metricsChart', plotData, layout, { responsive: true, displayModeBar: false });
}

window.switchTab = function(tab) {
    currentTab = tab;
    document.querySelectorAll('.chart-tab').forEach(b => b.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
    renderChart(allMetricsCache);
};

// ═══════════════════════════════════════════════════════════════
// Stats cards
// ═══════════════════════════════════════════════════════════════
function updateStatCards(d) {
    if (!d) return;
    const fmt = (v, dec=4) => (v != null) ? (+v).toFixed(dec) : '—';
    document.getElementById('sv-epoch').textContent = d.epoch ?? '—';
    document.getElementById('sv-iter').textContent  = d.iteration ?? '—';
    document.getElementById('sv-psnr').textContent  = fmt(d.psnr, 2);
    document.getElementById('sv-mae').textContent   = fmt(d.mae, 4);
    document.getElementById('sv-gloss').textContent = fmt(d.gen_loss, 4);
    document.getElementById('sv-dloss').textContent = fmt(d.dis_loss, 4);
    document.getElementById('sv-l1').textContent    = fmt(d.l1_loss, 4);
    document.getElementById('sv-perc').textContent  = fmt(d.perceptual_loss, 4);
}

// ═══════════════════════════════════════════════════════════════
// API fetchers
// ═══════════════════════════════════════════════════════════════
const connectionStatus = document.getElementById('connection-status');
const cmdBadge         = document.getElementById('current-command-badge');

async function fetchState() {
    try {
        const res  = await fetch(`${API_BASE}/command`);
        const data = await res.json();
        connectionStatus.className = 'pulse-dot online';
        cmdBadge.textContent  = data.command.toUpperCase();
        cmdBadge.className    = `badge badge-${data.command === 'run' ? 'active' : (data.command === 'stop' ? 'stopped' : 'pull')}`;
    } catch (e) {
        connectionStatus.className = 'pulse-dot';
    }
}

// Delta querying metrics
async function fetchAllMetrics() {
    if (selectedRun && allMetricsCache.length > 0) return;
    try {
        const afterIter = allMetricsCache.length > 0 ? allMetricsCache[allMetricsCache.length - 1].iteration : -1;
        const url = `${API_BASE}/all_metrics?after=${afterIter}${selectedRun ? '&run=' + selectedRun : ''}`;
        const res = await fetch(url);
        const data = await res.json();

        if (!Array.isArray(data) || data.length === 0) return;

        allMetricsCache = allMetricsCache.concat(data);

        // Update stat cards with latest data point
        updateStatCards(allMetricsCache[allMetricsCache.length - 1]);
        renderChart(allMetricsCache);
    } catch (e) { /* silent */ }
}

// Logs
const terminalOutput = document.getElementById('terminal-output');
async function fetchLogs() {
    if (selectedRun && lastLogCount > 0) return; // For archives, only fetch once
    try {
        const url = `${API_BASE}/logs?after=${lastLogCount}${selectedRun ? '&run=' + selectedRun : ''}`;
        const res = await fetch(url);
        const data = await res.json();

        // data is {lines: [], total: X} for live, or just [] for archive
        let newLines = [];
        if (selectedRun) {
            newLines = data;
            lastLogCount = newLines.length;
            terminalOutput.innerHTML = ''; // reset for archive
        } else {
            newLines = data.lines;
            if (data.total < lastLogCount) {
                // Server was reset or trimmed beyond our reach
                terminalOutput.innerHTML = '';
            }
            lastLogCount = data.total;
        }

        if (newLines.length > 0) {
            const html = newLines.map(l => `<div class="terminal-line">${formatLogLine(l)}</div>`).join('');
            terminalOutput.insertAdjacentHTML('beforeend', html);
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
            
            // Limit DOM size to last 2000 lines to prevent browser lag
            while (terminalOutput.children.length > 2000) {
                terminalOutput.removeChild(terminalOutput.firstChild);
            }
        }
    } catch (e) { /* silent */ }
}

function formatLogLine(line) {
    if (line.toLowerCase().includes('error'))   return `<span class="log-error">${line}</span>`;
    if (line.toLowerCase().includes('warning')) return `<span class="log-warn">${line}</span>`;
    if (line.includes('epoch') || line.includes('Epoch') || line.includes('[RESUME]'))
        return `<span class="log-info">${line}</span>`;
    if (line.includes('psnr') || line.includes('PSNR')) return `<span class="log-psnr">${line}</span>`;
    return line;
}

// Archive runs list
async function fetchRuns() {
    try {
        const runs = await (await fetch(`${API_BASE}/runs`)).json();
        const curr = runSelector.value;
        const opts = ['<option value="" style="background:#1e1e1e;">[ • Current Live Session ]</option>'];
        runs.forEach(r => opts.push(`<option value="${r}" style="background:#1e1e1e;">Archive: ${r}</option>`));
        runSelector.innerHTML = opts.join('');
        runSelector.value = curr;
    } catch (e) { /* silent */ }
}

runSelector.addEventListener('change', e => {
    selectedRun     = e.target.value;
    lastLogCount    = 0;
    allMetricsCache = [];
    progressionGroups = {};
    lastImageMeta = { count: -1, latest: null };
    Plotly.purge('metricsChart');
    terminalOutput.innerHTML = '';
    document.getElementById('image-gallery').innerHTML = '';
    fetchLogs(); fetchAllMetrics(); fetchImages();
});

// ═══════════════════════════════════════════════════════════════
// Images / Progression Viewer
// ═══════════════════════════════════════════════════════════════
const imageGallery   = document.getElementById('image-gallery');
const imageCount     = document.getElementById('image-count');
let progressionGroups = {};
let lastImageMeta    = { count: -1, latest: null };

// Memory management cache
let prefetchCache = new Map();
const PRELOAD_WINDOW = 10; // +- 10 images exactly as requested
let scrubTimeout = null;

function parseImageName(filename) {
    const match = filename.match(/^(.+?)_iter(\d+)\.(png|jpg|jpeg)$/i);
    if (match) return { base: match[1], iter: parseInt(match[2], 10) };
    return { base: filename.replace(/\.(png|jpg|jpeg)$/i, ''), iter: 0 };
}

function getAllIters() {
    const s = new Set();
    Object.values(progressionGroups).forEach(g => g.forEach(e => s.add(e.iter)));
    return Array.from(s).sort((a, b) => a - b);
}

async function fetchImages() {
    if (selectedRun && Object.keys(progressionGroups).length > 0) return;
    try {
        const metaRes = await fetch(`${API_BASE}/images/meta${selectedRun ? '?run=' + selectedRun : ''}`);
        const meta = await metaRes.json();
        if (meta.count === lastImageMeta.count && meta.latest === lastImageMeta.latest) return;
        lastImageMeta = meta;
        if (meta.count === 0) { imageGallery.innerHTML = ''; imageCount.textContent = '0 groups'; return; }
        const images = await (await fetch(`${API_BASE}/images${selectedRun ? '?run=' + selectedRun : ''}`)).json();
        if (images.length > 0) buildProgressionViewer(images);
    } catch (e) { /* silent */ }
}

function buildProgressionViewer(allImages) {
    progressionGroups = {};
    allImages.forEach(fn => {
        const { base, iter } = parseImageName(fn);
        if (!progressionGroups[base]) progressionGroups[base] = [];
        if (!progressionGroups[base].find(e => e.iter === iter))
            progressionGroups[base].push({ filename: fn, iter });
    });
    Object.keys(progressionGroups).forEach(b => progressionGroups[b].sort((a, b) => a.iter - b.iter));

    const allIters  = getAllIters();
    const numGroups = Object.keys(progressionGroups).length;
    imageCount.textContent = `${numGroups} image${numGroups !== 1 ? 's' : ''}`;
    if (!allIters.length) return;

    const scrubberBar = document.getElementById('scrubber-bar');
    const scrubber    = document.getElementById('global-scrubber');
    scrubberBar.style.display = 'block';
    if (scrubber.max == 0 || scrubber.max != allIters.length - 1) {
        scrubber.min = 0; scrubber.max = allIters.length - 1; 
    }
    
    document.getElementById('scrubber-min').textContent = `iter ${allIters[0]}`;
    document.getElementById('scrubber-max').textContent = `iter ${allIters[allIters.length - 1]}`;

    const prevIdx   = parseInt(scrubber.value, 10);
    const activeIdx = isNaN(prevIdx) ? allIters.length - 1 : Math.min(prevIdx, allIters.length - 1);
    scrubber.value  = activeIdx;

    const existing = imageGallery.querySelectorAll('.prog-card');
    if (existing.length !== numGroups || imageGallery.querySelector('.empty-state')) {
        imageGallery.innerHTML = '';
        Object.keys(progressionGroups).forEach(base => {
            const card = document.createElement('div');
            card.className = 'prog-card'; card.dataset.base = base; card.dataset.renderedIter = '';
            card.innerHTML = `
                <div class="prog-card-header" title="${base}">${base}</div>
                <img alt="${base}" onclick="openModal('${base}')" style="cursor:pointer;" title="Click for full evolution">
                <div class="prog-card-footer">
                    <span class="prog-iter-label">loading...</span>
                    <span class="prog-count-label">0 snapshots</span>
                </div>`;
            imageGallery.appendChild(card);
        });
    }

    // Direct invocation first load
    applyWindow(activeIdx, allIters);

    if (!scrubber._listenerAttached) {
        scrubber._listenerAttached = true;
        scrubber.addEventListener('input', () => {
            const iters = getAllIters(); if (!iters.length) return;
            const idx   = Math.min(parseInt(scrubber.value, 10), iters.length - 1);
            const iter  = iters[idx];
            document.getElementById('scrubber-current').textContent = `iter ${iter}`;
            document.getElementById('iter-badge').textContent = `iter ${iter}`;
            
            // Debounce load to save bandwidth and DOM CPU
            if (scrubTimeout) clearTimeout(scrubTimeout);
            scrubTimeout = setTimeout(() => {
                applyWindow(idx, iters);
            }, 1000); // Wait 1 second after scrubber pauses!
        });
    }
}

function applyWindow(centerIdx, allIters) {
    const targetIter = allIters[centerIdx];
    const loadStart = Math.max(0, centerIdx - PRELOAD_WINDOW);
    const loadEnd   = Math.min(allIters.length - 1, centerIdx + PRELOAD_WINDOW);

    // Track which images we want to keep in the browser cache
    const urlsToKeep = new Set();

    imageGallery.querySelectorAll('.prog-card').forEach(card => {
        const base  = card.dataset.base;
        const group = progressionGroups[base]; if (!group) return;
        const img   = card.querySelector('img');
        const iterL = card.querySelector('.prog-iter-label');
        const cntL  = card.querySelector('.prog-count-label');

        let best = null;
        for (const e of group) { if (e.iter <= targetIter) best = e; else break; }
        
        if (best) {
            if (card.dataset.renderedIter !== String(best.iter)) {
                card.dataset.renderedIter = String(best.iter);
                img.classList.add('fading');
                setTimeout(() => {
                    img.src = selectedRun ? `/images_archive/${selectedRun}/${best.filename}` : `/images/${best.filename}`;
                    img.classList.remove('fading');
                }, 120);
                iterL.textContent = `iter ${best.iter}`;
            }
        } else {
            img.removeAttribute('src'); iterL.textContent = 'no snapshot yet';
        }
        cntL.textContent = `${group.length} snapshot${group.length !== 1 ? 's' : ''}`;

        // Prepare the 20 prefetch Image objects in memory
        for (let i = loadStart; i <= loadEnd; i++) {
            const iv = allIters[i];
            let prev = null;
            for (const e of group) { if (e.iter <= iv) prev = e; else break; }
            if (prev) {
                const url = selectedRun ? `/images_archive/${selectedRun}/${prev.filename}` : `/images/${prev.filename}`;
                urlsToKeep.add(url);
                if (!prefetchCache.has(url)) {
                    const iObj = new Image(); iObj.src = url;
                    prefetchCache.set(url, iObj);
                }
            }
        }
    });

    // Cleanup memory: destroy Image objects older than the window
    for (const [url, obj] of prefetchCache.entries()) {
        if (!urlsToKeep.has(url)) {
            prefetchCache.delete(url);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Controls
// ═══════════════════════════════════════════════════════════════
async function sendCommand(cmd) {
    await fetch(`${API_BASE}/command`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd })
    });
    fetchState();
}

const terminalInput = document.getElementById('terminal-input');

async function sendShellCommand(shellCmd) {
    if (!shellCmd.trim()) return;
    await fetch(`${API_BASE}/command`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ shell_command: shellCmd })
    });
    terminalInput.value = '';
}

document.getElementById('btn-run').addEventListener('click', () => {
    if (!selectedRun && confirm('Start training?')) sendCommand('run');
    else if (selectedRun) alert('Return to Live Session to run training.');
});

document.getElementById('btn-stop').addEventListener('click', async () => {
    if (!selectedRun && confirm('⚠️ Stop training?')) {
        await sendCommand('stop');
        const name = prompt('Save run as:', `Run_${new Date().toISOString().slice(0,10).replace(/-/g,'')}`);
        if (name) {
            await fetch(`${API_BASE}/save_run`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            await fetchRuns();
            runSelector.value = name;
            runSelector.dispatchEvent(new Event('change'));
        }
    }
});

document.getElementById('btn-pull').addEventListener('click', () => {
    if (confirm('Git Pull & Restart?')) sendCommand('restart_pull');
});

document.getElementById('btn-send-cmd').addEventListener('click', () => sendShellCommand(terminalInput.value));
terminalInput.addEventListener('keypress', e => { if (e.key === 'Enter') sendShellCommand(terminalInput.value); });
document.getElementById('smooth-window').addEventListener('input', () => renderChart(allMetricsCache));

// ═══════════════════════════════════════════════════════════════
// Modal
// ═══════════════════════════════════════════════════════════════
let isModalOpen = false, modalBaseName = null, modalIterIdx = 0;

window.openModal = function(base) {
    if (!progressionGroups[base]) return;
    modalBaseName = base; modalIterIdx = progressionGroups[base].length - 1;
    isModalOpen = true;
    document.getElementById('evolution-modal').style.display = 'flex';
    document.getElementById('modal-basename').textContent = base;
    updateModalImage();
};

window.closeModal = function() {
    isModalOpen = false;
    document.getElementById('evolution-modal').style.display = 'none';
};

window.modalStep = function(dir) { modalIterIdx += dir; updateModalImage(); };

function updateModalImage() {
    if (!isModalOpen || !modalBaseName) return;
    const group = progressionGroups[modalBaseName]; if (!group || !group.length) return;
    modalIterIdx = Math.max(0, Math.min(modalIterIdx, group.length - 1));
    const e = group[modalIterIdx];
    document.getElementById('modal-img').src = selectedRun ? `/images_archive/${selectedRun}/${e.filename}` : `/images/${e.filename}`;
    document.getElementById('modal-iter').textContent    = `Iteration: ${e.iter}`;
    document.getElementById('modal-counter').textContent = `${modalIterIdx + 1} / ${group.length}`;
}

document.addEventListener('keydown', e => {
    if (!isModalOpen) return;
    if (e.key === 'Escape') closeModal();
    if (e.key === 'ArrowLeft')  modalStep(-1);
    if (e.key === 'ArrowRight') modalStep(1);
});

// ═══════════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════════
fetchRuns();
fetchLogs();
fetchAllMetrics();
fetchImages();
fetchState();

setInterval(fetchState,      1000);
setInterval(fetchLogs,       1000);
setInterval(fetchAllMetrics, 2000);
setInterval(fetchImages,     5000);

