const API_BASE = '/api';

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════
let selectedRun = '';
let allMetricsCache = [];   // full structured metrics for current view
let currentTab = 'losses';  // 'losses' | 'quality'

const runSelector = document.getElementById('run-selector');

// ═══════════════════════════════════════════════════════════════
// Chart colours + dataset definitions
// ═══════════════════════════════════════════════════════════════
const LOSS_DATASETS = [
    { key: 'gen_loss',        label: 'Gen Loss (total)', color: '#a855f7', yAxis: 'y' },
    { key: 'dis_loss',        label: 'Dis Loss',         color: '#ef4444', yAxis: 'y' },
    { key: 'l1_loss',         label: 'L1 Loss',          color: '#3b82f6', yAxis: 'y' },
    { key: 'perceptual_loss', label: 'Perceptual',       color: '#10b981', yAxis: 'y' },
    { key: 'style_loss',      label: 'Style',            color: '#f59e0b', yAxis: 'y' },
    { key: 'sym_loss',        label: 'Symmetry',         color: '#64748b', yAxis: 'y' },
];

const QUALITY_DATASETS = [
    { key: 'psnr', label: 'PSNR (dB)', color: '#d8b4fe', yAxis: 'y'  },
    { key: 'mae',  label: 'MAE',       color: '#fb923c', yAxis: 'y2' },
];

// ═══════════════════════════════════════════════════════════════
// Chart
// ═══════════════════════════════════════════════════════════════
let metricsChart;

function makeDatasets(defs) {
    return defs.map(d => ({
        label: d.label,
        data: [],
        borderColor: d.color,
        backgroundColor: d.color + '22',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 0,
        yAxisID: d.yAxis,
        _key: d.key,
    }));
}

function initChart() {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: makeDatasets(LOSS_DATASETS) },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8', font: { family: 'Outfit' }, boxWidth: 12 }
                },
                tooltip: { backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f8fafc', bodyColor: '#94a3b8' }
            },
            scales: {
                y:  { type: 'linear', position: 'left',  grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                y2: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#fb923c' }, display: false },
                x:  { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', maxTicksLimit: 15, font: { size: 10 } } },
            },
            animation: { duration: 0 },
        }
    });
}

window.switchTab = function(tab) {
    currentTab = tab;
    document.querySelectorAll('.chart-tab').forEach(b => b.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');

    const defs = tab === 'losses' ? LOSS_DATASETS : QUALITY_DATASETS;
    metricsChart.data.datasets = makeDatasets(defs);
    metricsChart.data.labels = [];

    // Toggle y2 axis visibility (only for quality tab with MAE)
    metricsChart.options.scales.y2.display = (tab === 'quality');
    metricsChart.update();

    renderChart(allMetricsCache);
};

// Apply moving-average smoothing
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

function renderChart(data) {
    if (!metricsChart || data.length === 0) return;
    const w = Math.max(1, parseInt(document.getElementById('smooth-window').value) || 10);
    const defs = currentTab === 'losses' ? LOSS_DATASETS : QUALITY_DATASETS;

    const labels = data.map(d => `${d.iteration}`);
    metricsChart.data.labels = labels;

    metricsChart.data.datasets.forEach((ds, i) => {
        const key = defs[i].key;
        const raw = data.map(d => (d[key] != null ? d[key] : null));
        ds.data = smooth(raw.filter(v => v !== null), w);
        // Re-pad if any nulls (simple: just use filtered; length may differ — keep raw for now)
        ds.data = smooth(raw.map(v => v ?? 0), w);
    });

    metricsChart.update();
}

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

// All structured metrics
let lastMetricsLen = -1;
async function fetchAllMetrics() {
    // Freeze polling if we're in an archived run and already have data
    if (selectedRun && allMetricsCache.length > 0) return;
    try {
        // Thin by 2 for large sessions to keep payload sane
        const thin = allMetricsCache.length > 5000 ? 2 : 1;
        const url  = `${API_BASE}/all_metrics?thin=${thin}${selectedRun ? '&run=' + selectedRun : ''}`;
        const res  = await fetch(url);
        const data = await res.json();

        if (!Array.isArray(data) || data.length === 0) return;
        if (data.length === lastMetricsLen) return; // no change

        lastMetricsLen  = data.length;
        allMetricsCache = data;

        // Update stat cards with latest data point
        updateStatCards(data[data.length - 1]);
        renderChart(data);
    } catch (e) { /* silent */ }
}

// Logs
let lastLogHash = '';
const terminalOutput = document.getElementById('terminal-output');
async function fetchLogs() {
    if (selectedRun && lastLogHash !== '') return;
    try {
        const res   = await fetch(`${API_BASE}/logs${selectedRun ? '?run=' + selectedRun : ''}`);
        const lines = await res.json();
        if (lines.length > 0) {
            const hash = lines[lines.length - 1];
            if (hash !== lastLogHash) {
                lastLogHash = hash;
                terminalOutput.innerHTML = lines.map(l => `<div class="terminal-line">${formatLogLine(l)}</div>`).join('');
                terminalOutput.scrollTop = terminalOutput.scrollHeight;
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
    lastLogHash     = '';
    lastMetricsLen  = -1;
    allMetricsCache = [];
    progressionGroups = {};
    lastImageMeta = { count: -1, latest: null };
    if (metricsChart) { metricsChart.data.labels = []; metricsChart.data.datasets.forEach(d => d.data = []); metricsChart.update(); }
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
const prefetchedUrls = new Set();
const PRELOAD_AHEAD  = 5;
const PRELOAD_BEHIND = 3;

function prefetchUrl(url) {
    if (prefetchedUrls.has(url)) return;
    prefetchedUrls.add(url);
    const img = new Image(); img.src = url;
}

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
    scrubber.min = 0; scrubber.max = allIters.length - 1;
    document.getElementById('scrubber-min').textContent = `iter ${allIters[0]}`;
    document.getElementById('scrubber-max').textContent = `iter ${allIters[allIters.length - 1]}`;

    const prevIdx   = parseInt(scrubber.value, 10);
    const activeIdx = isNaN(prevIdx) ? allIters.length - 1 : Math.min(prevIdx, allIters.length - 1);
    scrubber.value  = activeIdx;
    const activeIter = allIters[activeIdx];
    document.getElementById('scrubber-current').textContent = `iter ${activeIter}`;
    document.getElementById('iter-badge').textContent = `iter ${activeIter}`;

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

    applyWindow(activeIdx, allIters);

    if (!scrubber._listenerAttached) {
        scrubber._listenerAttached = true;
        scrubber.addEventListener('input', () => {
            const iters = getAllIters(); if (!iters.length) return;
            const idx   = Math.min(parseInt(scrubber.value, 10), iters.length - 1);
            const iter  = iters[idx];
            document.getElementById('scrubber-current').textContent = `iter ${iter}`;
            document.getElementById('iter-badge').textContent = `iter ${iter}`;
            applyWindow(idx, iters);
        });
    }
}

function applyWindow(centerIdx, allIters) {
    const loadStart = Math.max(0, centerIdx - PRELOAD_BEHIND);
    const loadEnd   = Math.min(allIters.length - 1, centerIdx + PRELOAD_AHEAD);
    const targetIter = allIters[centerIdx];

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

        for (let i = loadStart; i <= loadEnd; i++) {
            const iv = allIters[i];
            let prev = null;
            for (const e of group) { if (e.iter <= iv) prev = e; else break; }
            if (prev) prefetchUrl(`/images/${prev.filename}`);
        }
    });
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
initChart();
fetchRuns();

setInterval(fetchState,      1000);
setInterval(fetchLogs,       1000);
setInterval(fetchAllMetrics, 2000);
setInterval(fetchImages,     5000);
