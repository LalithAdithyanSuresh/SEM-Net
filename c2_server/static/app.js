const API_BASE = '/api';

// Run Selection State
let selectedRun = '';
const runSelector = document.getElementById('run-selector');

async function fetchRuns() {
    try {
        const res = await fetch(`${API_BASE}/runs`);
        const runs = await res.json();
        
        // Preserve current selection if it still exists
        const currentVal = runSelector.value;
        const opts = ['<option value="" style="background: #1e1e1e;">[ • Current Live Session ]</option>'];
        runs.forEach(r => {
            opts.push(`<option value="${r}" style="background: #1e1e1e;">Archive: ${r}</option>`);
        });
        runSelector.innerHTML = opts.join('');
        runSelector.value = currentVal;
    } catch (e) { console.error("Failed to fetch runs:", e); }
}

runSelector.addEventListener('change', (e) => {
    selectedRun = e.target.value;
    
    // Clear State for clean slate
    if(metricsChart) {
        metricsChart.data.labels = [];
        metricsChart.data.datasets.forEach(d => d.data = []);
        metricsChart.update();
    }
    terminalOutput.innerHTML = '';
    imageGallery.innerHTML = '';
    imageCount.textContent = '0 images';
    progressionGroups = {};
    lastLogHash = "";
    lastImageMeta = { count: -1, latest: null };
    
    fetchLogs();
    fetchMetrics();
    fetchImages();
});

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
                { label: 'L1 Loss', data: [], borderColor: '#3b82f6', tension: 0.4, borderWidth: 2, yAxisID: 'y' },
                { label: 'Perceptual', data: [], borderColor: '#10b981', tension: 0.4, borderWidth: 2, yAxisID: 'y' },
                { label: 'Adversarial', data: [], borderColor: '#f59e0b', tension: 0.4, borderWidth: 2, yAxisID: 'y' },
                { label: 'PSNR', data: [], borderColor: '#d8b4fe', tension: 0.4, borderWidth: 2, yAxisID: 'y1' } // PSNR
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { type: 'linear', position: 'left', grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#d8b4fe' } },
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
    if (selectedRun && metricsChart && metricsChart.data.labels.length > 0) return; // Freeze poll if archive is loaded
    try {
        const res = await fetch(`${API_BASE}/metrics${selectedRun ? '?run='+selectedRun : ''}`);
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

let lastLogHash = "";
let psnrHistory = [];
// Fetch Logs
async function fetchLogs() {
    if (selectedRun && lastLogHash !== "") return; // Freeze poll if archive is loaded
    try {
        const res = await fetch(`${API_BASE}/logs${selectedRun ? '?run='+selectedRun : ''}`);
        const lines = await res.json();
        
        if (lines.length > 0) {
            // Fix: Hash based on the exact final string to prevent arbitrary 500-line limit freeze
            const currentHash = lines[lines.length - 1];
            if (currentHash !== lastLogHash) {
                lastLogHash = currentHash;
                terminalOutput.innerHTML = lines.map(l => `<div class="terminal-line">${formatLogLine(l)}</div>`).join('');
                terminalOutput.scrollTop = terminalOutput.scrollHeight;
                
                // Parse PSNR via Regex into the chart
                let newPsnrData = [];
                lines.forEach(l => {
                    const match = l.match(/psnr:\s*([\d\.]+)/i);
                    if (match) newPsnrData.push(parseFloat(match[1]));
                });
                
                if (newPsnrData.length > 0 && metricsChart) {
                    const latestPSNR = newPsnrData[newPsnrData.length - 1];
                    // Link the latest PSNR to the current length of the labels
                    const numLabels = metricsChart.data.labels.length;
                    let psnrDataset = metricsChart.data.datasets[3].data;
                    if (numLabels > 0 && psnrDataset.length < numLabels) {
                        while (psnrDataset.length < numLabels - 1) psnrDataset.push(psnrDataset[psnrDataset.length - 1] || latestPSNR);
                        psnrDataset[numLabels - 1] = latestPSNR;
                        metricsChart.update();
                    }
                }
            }
        }
    } catch (e) { }
}

// Efficient image fetching: poll a lightweight /meta endpoint first,
// only fetch the full image list when count or latest filename changes.
let lastImageMeta = { count: -1, latest: null };

async function fetchImages() {
    if (selectedRun && Object.keys(progressionGroups).length > 0) return; // Freeze poll if archive is loaded
    try {
        // Step 1: cheap meta check (~50 bytes)
        const metaRes = await fetch(`${API_BASE}/images/meta${selectedRun ? '?run='+selectedRun : ''}`);
        const meta = await metaRes.json();

        // Step 2: bail out early if nothing changed
        if (meta.count === lastImageMeta.count && meta.latest === lastImageMeta.latest) {
            return;
        }
        lastImageMeta = meta;

        // Step 3: only NOW fetch the full list (triggered only on actual change)
        if (meta.count === 0) {
            imageGallery.innerHTML = '';
            imageCount.textContent = '0 images';
            return;
        }
        const res = await fetch(`${API_BASE}/images${selectedRun ? '?run='+selectedRun : ''}`);
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
                <img alt="${base}" onclick="openModal('${base}')" style="cursor: pointer;" title="Click to view full evolution">
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
                    img.src = selectedRun ? `/images_archive/${selectedRun}/${best.filename}` : `/images/${best.filename}`;
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

btnRun.addEventListener('click', () => { 
    if(!selectedRun && confirm('Start training run?')) sendCommand('run');
    else if(selectedRun) alert("Please return to Current Live Session to run training!");
});

btnStop.addEventListener('click', async () => { 
    if(!selectedRun && confirm('⚠️ Stop training?')) {
        await sendCommand('stop');
        const rName = prompt("Training stopped. Enter a name to save this run:", `Run_${new Date().toISOString().slice(0,10).replace(/-/g, '')}`);
        if(rName) {
            await fetch(`${API_BASE}/save_run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: rName })
            });
            await fetchRuns();
            runSelector.value = rName;
            runSelector.dispatchEvent(new Event('change'));
        }
    }
});
btnPull.addEventListener('click', () => { if(confirm('Git Pull and Restart?')) sendCommand('restart_pull') });

btnSendCmd.addEventListener('click', () => sendShellCommand(terminalInput.value));
terminalInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendShellCommand(terminalInput.value);
});

// ================================================================
// Gallery Modal Evolution Viewer
// ================================================================
let isModalOpen = false;
let modalBaseName = null;
let modalIterIdx = 0;

window.openModal = function(baseName) {
    if (!progressionGroups[baseName]) return;
    modalBaseName = baseName;
    modalIterIdx = progressionGroups[baseName].length - 1; // Start at latest
    isModalOpen = true;
    document.getElementById('evolution-modal').style.display = 'flex';
    document.getElementById('modal-basename').textContent = baseName;
    updateModalImage();
};

window.closeModal = function() {
    isModalOpen = false;
    document.getElementById('evolution-modal').style.display = 'none';
};

window.modalStep = function(dir) {
    modalIterIdx += dir;
    updateModalImage();
};

function updateModalImage() {
    if (!isModalOpen || !modalBaseName) return;
    const group = progressionGroups[modalBaseName];
    if (!group || group.length === 0) return;
    
    // Clamp
    if (modalIterIdx < 0) modalIterIdx = 0;
    if (modalIterIdx >= group.length) modalIterIdx = group.length - 1;
    
    const entry = group[modalIterIdx];
    document.getElementById('modal-img').src = selectedRun ? `/images_archive/${selectedRun}/${entry.filename}` : `/images/${entry.filename}`;
    document.getElementById('modal-iter').textContent = `Iteration: ${entry.iter}`;
    document.getElementById('modal-counter').textContent = `${modalIterIdx + 1} / ${group.length}`;
}

// Keyboard nav
document.addEventListener('keydown', (e) => {
    if (!isModalOpen) return;
    if (e.key === 'Escape') closeModal();
    if (e.key === 'ArrowLeft') { modalStep(-1); }
    if (e.key === 'ArrowRight') { modalStep(1); }
});

// Bootstrap
initChart();
fetchRuns();

// Polling Loops
setInterval(fetchState, 1000);
setInterval(fetchLogs, 1000);
setInterval(fetchMetrics, 3000);
setInterval(fetchImages, 5000);

