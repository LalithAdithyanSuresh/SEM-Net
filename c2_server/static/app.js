const API_BASE = '/api';

// UI Elements
const btnRun = document.getElementById('btn-run');
const btnStop = document.getElementById('btn-stop');
const btnPull = document.getElementById('btn-pull');
const cmdBadge = document.getElementById('current-command-badge');
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

    } catch (e) {
        connectionStatus.className = 'pulse-dot'; // Offline
        console.error('Failed to fetch state');
    }
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
    } catch (e) { }
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
    } catch (e) { }
}

// Efficient image fetching: poll a lightweight /meta endpoint first,
// only fetch the full image list when count or latest filename changes.
let lastImageMeta = { count: -1, latest: null };

async function fetchImages() {
    try {
        // Step 1: cheap meta check (~50 bytes)
        const metaRes = await fetch(`${API_BASE}/images/meta`);
        const meta = await metaRes.json();

        // Step 2: bail out early if nothing changed
        if (meta.count === lastImageMeta.count && meta.latest === lastImageMeta.latest) {
            return;
        }
        lastImageMeta = meta;

        // Step 3: only NOW fetch the full list (triggered only on actual change)
        if (meta.count === 0) return;
        const res = await fetch(`${API_BASE}/images`);
        const images = await res.json();
        if (images.length > 0) buildProgressionViewer(images);
    } catch (e) { }
}

// ================================================================
// Progression Viewer — Windowed Loading
// Only images within ±PRELOAD_AHEAD/BEHIND of current slider
// position ever have img.src set. Everything else is unloaded.
// ================================================================

// Tuning constants
const PRELOAD_AHEAD = 5;   // load up to 5 iters forward from current
const PRELOAD_BEHIND = 3;   // keep 3 iters behind current loaded
const UNLOAD_DIST = 20;  // unload anything futher than ±10 iters away

// In-memory metadata: { baseName -> [ {filename, iter} ] }
let progressionGroups = {};

// Browser-cache warmer: tracks which filenames have been prefetched
// so we never kick off the same request twice.
const prefetchedUrls = new Set();

function prefetchUrl(url) {
    if (prefetchedUrls.has(url)) return;
    prefetchedUrls.add(url);
    const img = new Image();
    img.src = url;   // fires the HTTP request, browser caches the result
}

function parseImageName(filename) {
    const match = filename.match(/^(.+?)_iter(\d+)\.(png|jpg|jpeg)$/i);
    if (match) return { base: match[1], iter: parseInt(match[2], 10) };
    return { base: filename.replace(/\.(png|jpg|jpeg)$/i, ''), iter: 0 };
}

function getAllIters() {
    const iterSet = new Set();
    Object.values(progressionGroups).forEach(g => g.forEach(e => iterSet.add(e.iter)));
    return Array.from(iterSet).sort((a, b) => a - b);
}

// Returns the index of targetIter in allIters (exact match)
function iterToIdx(allIters, targetIter) {
    return allIters.indexOf(targetIter);
}

function buildProgressionViewer(allImages) {
    // Rebuild metadata (just strings — no network cost)
    progressionGroups = {};
    allImages.forEach(filename => {
        const { base, iter } = parseImageName(filename);
        if (!progressionGroups[base]) progressionGroups[base] = [];
        if (!progressionGroups[base].find(e => e.iter === iter)) {
            progressionGroups[base].push({ filename, iter });
        }
    });
    Object.keys(progressionGroups).forEach(b => {
        progressionGroups[b].sort((a, b) => a.iter - b.iter);
    });

    const allIters = getAllIters();
    const numGroups = Object.keys(progressionGroups).length;
    imageCount.textContent = `${numGroups} image${numGroups !== 1 ? 's' : ''}`;
    if (allIters.length === 0) return;

    // Configure the scrubber
    const scrubberBar = document.getElementById('scrubber-bar');
    const scrubber = document.getElementById('global-scrubber');
    const scrubMin = document.getElementById('scrubber-min');
    const scrubMax = document.getElementById('scrubber-max');
    const scrubCurrent = document.getElementById('scrubber-current');
    const iterBadge = document.getElementById('iter-badge');

    scrubberBar.style.display = 'block';
    scrubber.min = 0;
    scrubber.max = allIters.length - 1;
    scrubMin.textContent = `iter ${allIters[0]}`;
    scrubMax.textContent = `iter ${allIters[allIters.length - 1]}`;

    // On first build jump to latest; on subsequent polls preserve position
    const prevIdx = parseInt(scrubber.value, 10);
    const activeIdx = isNaN(prevIdx) ? allIters.length - 1
        : Math.min(prevIdx, allIters.length - 1);
    scrubber.value = activeIdx;
    const activeIter = allIters[activeIdx];
    scrubCurrent.textContent = `iter ${activeIter}`;
    iterBadge.textContent = `iter ${activeIter}`;

    // Build DOM cards (only once)
    const gallery = imageGallery;
    const existingCards = gallery.querySelectorAll('.prog-card');
    if (existingCards.length !== numGroups || gallery.querySelector('.empty-state')) {
        gallery.innerHTML = '';
        Object.keys(progressionGroups).forEach(base => {
            const card = document.createElement('div');
            card.className = 'prog-card';
            card.dataset.base = base;
            card.dataset.renderedIter = '';
            card.innerHTML = `
                <div class="prog-card-header" title="${base}">${base}</div>
                <img alt="${base}">
                <div class="prog-card-footer">
                    <span class="prog-iter-label">loading...</span>
                    <span class="prog-count-label">0 snapshots</span>
                </div>`;
            gallery.appendChild(card);
        });
    }

    // Show the window for the current position
    applyWindow(activeIdx, allIters);

    // Attach scrubber listener once
    if (!scrubber._listenerAttached) {
        scrubber._listenerAttached = true;
        scrubber.addEventListener('input', () => {
            const idx = parseInt(scrubber.value, 10);
            const iters = getAllIters();
            if (!iters.length) return;
            const clamped = Math.min(idx, iters.length - 1);
            const iter = iters[clamped];
            scrubCurrent.textContent = `iter ${iter}`;
            iterBadge.textContent = `iter ${iter}`;
            applyWindow(clamped, iters);
        });
    }
}

// ---- Core windowed logic ----
function applyWindow(centerIdx, allIters) {
    const loadStart = Math.max(0, centerIdx - PRELOAD_BEHIND);
    const loadEnd = Math.min(allIters.length - 1, centerIdx + PRELOAD_AHEAD);
    const unloadStart = Math.max(0, centerIdx - UNLOAD_DIST);
    const unloadEnd = Math.min(allIters.length - 1, centerIdx + UNLOAD_DIST);

    const targetIter = allIters[centerIdx];

    const cards = imageGallery.querySelectorAll('.prog-card');
    cards.forEach(card => {
        const base = card.dataset.base;
        const group = progressionGroups[base];
        if (!group) return;

        const img = card.querySelector('img');
        const iterLabel = card.querySelector('.prog-iter-label');
        const countLabel = card.querySelector('.prog-count-label');

        // --- 1. Display image: best snapshot <= targetIter within load window ---
        let best = null;
        for (const entry of group) {
            if (entry.iter <= targetIter) best = entry;
            else break;
        }

        if (best) {
            if (card.dataset.renderedIter !== String(best.iter)) {
                card.dataset.renderedIter = String(best.iter);
                img.classList.add('fading');
                setTimeout(() => {
                    img.src = `/images/${best.filename}`;
                    img.classList.remove('fading');
                }, 120);
                iterLabel.textContent = `iter ${best.iter}`;
            }
        } else {
            if (card.dataset.renderedIter !== 'none') {
                card.dataset.renderedIter = 'none';
                img.removeAttribute('src');
                iterLabel.textContent = 'no snapshot yet';
            }
        }

        countLabel.textContent = `${group.length} snapshot${group.length !== 1 ? 's' : ''}`;

        // --- 2. Silent prefetch: warm browser cache for ±PRELOAD_AHEAD iters ---
        for (let i = loadStart; i <= loadEnd; i++) {
            const iterVal = allIters[i];
            // Find closest entry in this group for that iter
            let prev = null;
            for (const entry of group) {
                if (entry.iter <= iterVal) prev = entry;
                else break;
            }
            if (prev) prefetchUrl(`/images/${prev.filename}`);
        }

        // --- 3. Unload: clear src for images far outside the window ---
        for (let i = 0; i < unloadStart; i++) {
            // nothing to unload in DOM (we only set src for displayed iter)
            // just ensure the rendered state is cleared if card was there
        }
        // The card img always shows only ONE src at a time —
        // so unloading is automatic: as we move the slider,
        // the old src is replaced and the browser GCs the old blob.
    });
}

// Actions
async function sendCommand(cmd) {
    await fetch(`${API_BASE}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd })
    });
    fetchState();
}

const terminalInput = document.getElementById('terminal-input');
const btnSendCmd = document.getElementById('btn-send-cmd');

async function sendShellCommand(shellCmd) {
    if (!shellCmd.trim()) return;

    await fetch(`${API_BASE}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ shell_command: shellCmd })
    });

    terminalInput.value = '';
}

btnRun.addEventListener('click', () => sendCommand('run'));
btnStop.addEventListener('click', () => sendCommand('stop'));
btnPull.addEventListener('click', () => sendCommand('restart_pull'));

btnSendCmd.addEventListener('click', () => sendShellCommand(terminalInput.value));
terminalInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendShellCommand(terminalInput.value);
});

// Bootstrap
initChart();

// Polling Loops
setInterval(fetchState, 1000);
setInterval(fetchLogs, 1000);
setInterval(fetchMetrics, 3000);
setInterval(fetchImages, 5000);

