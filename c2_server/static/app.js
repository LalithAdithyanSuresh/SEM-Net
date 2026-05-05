const API_BASE = '/api';

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════
let selectedRun = '';
let allMetricsCache = [];   // full structured metrics for current view
let lastLogCount = 0;       // tracks total lines received for delta fetching
let currentTab = 'quality'; // 'losses' | 'quality'

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
    { key: 'mae',             label: 'MAE',              color: '#ffffff', axis: 'y2' }, // crisp white
];

const QUALITY_DATASETS = [
    { key: 'psnr', label: 'PSNR (dB)', color: '#00ff88', axis: 'y'  }, // neon green
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
    
    let filteredData = data;
    const hideLast = document.getElementById('toggle-filter')?.checked;
    if (hideLast) {
        const kHide = parseFloat(document.getElementById('filter-k')?.value || 28);
        if (filteredData.length > 0) {
            const maxIter = filteredData[filteredData.length - 1].iteration;
            const threshold = maxIter - (kHide * 1000);
            filteredData = filteredData.filter(d => d.iteration >= threshold);
        }
    }
    if (filteredData.length === 0) return;

    const w = Math.max(1, parseInt(document.getElementById('smooth-window').value) || 10);
    const defs = currentTab === 'losses' ? LOSS_DATASETS : QUALITY_DATASETS;

    const showRoC = document.getElementById('toggle-roc')?.checked;
    const showPredict = document.getElementById('toggle-predict')?.checked;
    const predictM = parseFloat(document.getElementById('predict-m')?.value || 0);

    const iterations = filteredData.map(d => d.iteration);
    
    let plotData = [];

    defs.forEach(d => {
        const raw = filteredData.map(pt => pt[d.key] != null ? pt[d.key] : null);
        const smoothed = smooth(raw.map(v => v ?? 0), w);
        let isMapped = currentTab === 'quality' && d.key === 'psnr';
        let mappedY = smoothed;

        if (isMapped) {
            mappedY = smoothed.map(y => {
                if (y == null) return null;
                if (y <= 25) return y;
                return 25 + (y - 25) * 5; // 5x stretch above 25
            });
        }
        
        let trace = {
            x: iterations,
            y: mappedY,
            name: d.label,
            type: 'scatter',
            mode: 'lines',
            line: { color: d.color, width: 2.5 },
            fill: 'tozeroy',
            fillcolor: d.color + '22',
            yaxis: d.axis
        };

        if (isMapped) {
            trace.customdata = smoothed;
            trace.hovertemplate = '%{customdata:.2f}';
        }

        plotData.push(trace);

        if (showRoC && d.key === 'psnr') {
            let roc = [0];
            for (let i = 1; i < smoothed.length; i++) {
                let dx = iterations[i] - iterations[i-1];
                let dy = smoothed[i] - smoothed[i-1];
                roc.push(dx === 0 ? 0 : dy / dx * 1000); // ROC per 1k iters
            }
            roc = smooth(roc, w);
            plotData.push({
                x: iterations,
                y: roc,
                name: 'PSNR RoC (/1k iters)',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#ff2d78', width: 2, dash: 'dot' },
                yaxis: 'y2'
            });
        }

        if (!showPredict) {
            const b = document.getElementById('predict-coverage');
            if (b) b.style.display = 'none';
        }

        if (showPredict) {
            // Use at minimum last 50k iters worth of data, or 30% of data (whichever is more)
            const FIT_ITERS = 50000;
            let last_x = iterations[iterations.length - 1];
            let next_x = last_x + (predictM * 1000);

            let fullPrior = document.getElementById('toggle-full-prior')?.checked;
            let fitStart = 0;
            if (!fullPrior) {
                for (let i = iterations.length - 1; i >= 0; i--) {
                    if (last_x - iterations[i] >= FIT_ITERS) { fitStart = i; break; }
                }
                let n30 = Math.floor(smoothed.length * 0.3);
                fitStart = Math.min(fitStart, smoothed.length - n30);
                fitStart = Math.max(0, fitStart);
            }

            let x_sl = iterations.slice(fitStart);
            let y_sl = smoothed.slice(fitStart);
            let n = x_sl.length;

            // Update prior coverage badge
            let coveragePct = Math.round((n / iterations.length) * 100);
            let badge = document.getElementById('predict-coverage');
            if (badge) {
                badge.style.display = 'inline';
                badge.textContent = `Prior: ${coveragePct}%`;
                badge.style.color = coveragePct >= 90 ? '#00ff88'
                                  : coveragePct >= 60 ? '#ffe600'
                                  : '#ff6b00';
                badge.title = `Using ${n} of ${iterations.length} data points (${coveragePct}%) for model fit`;
            }
            if (n >= 4) {
                // Fit: y = a * log(x - x0) + b  where x0 = x_sl[0] - 1
                // Transform: t = log(x - x0 + 1)
                let x0 = x_sl[0] - 1;
                let t_sl = x_sl.map(x => Math.log(x - x0 + 1));
                let sum_t = 0, sum_y = 0, sum_tt = 0, sum_ty = 0;
                for (let i = 0; i < n; i++) {
                    sum_t  += t_sl[i];
                    sum_y  += y_sl[i];
                    sum_tt += t_sl[i] * t_sl[i];
                    sum_ty += t_sl[i] * y_sl[i];
                }
                let denom = n * sum_tt - sum_t * sum_t;
                let a = denom !== 0 ? (n * sum_ty - sum_t * sum_y) / denom : 0;
                let b = (sum_y - a * sum_t) / n;

                // Also compute linear fit for blending (handles decelerating / still-rising)
                let sum_x2 = 0, sum_xy2 = 0;
                for (let i = 0; i < n; i++) {
                    sum_x2  += x_sl[i];
                    sum_xy2 += x_sl[i] * y_sl[i];
                }
                let lin_denom = n * x_sl.reduce((acc, v) => acc + v*v, 0) - sum_x2 * sum_x2;
                let lin_slope = lin_denom !== 0 ? (n * sum_xy2 - sum_x2 * sum_y) / lin_denom : 0;
                let lin_int   = (sum_y - lin_slope * sum_x2) / n;

                // Pick model with lower residual on fit window
                let res_log = 0, res_lin = 0;
                for (let i = 0; i < n; i++) {
                    let yhat_log = a * t_sl[i] + b;
                    let yhat_lin = lin_slope * x_sl[i] + lin_int;
                    res_log += (y_sl[i] - yhat_log) ** 2;
                    res_lin += (y_sl[i] - yhat_lin) ** 2;
                }
                let useLog = res_log <= res_lin;

                // Generate 30 evenly spaced prediction points from last_x → next_x
                const PRED_POINTS = 30;
                let pred_xs = [], pred_ys = [], pred_mapped = [];
                for (let i = 0; i <= PRED_POINTS; i++) {
                    let xi = last_x + (i / PRED_POINTS) * (next_x - last_x);
                    let yi;
                    if (useLog) {
                        yi = a * Math.log(xi - x0 + 1) + b;
                    } else {
                        yi = lin_slope * xi + lin_int;
                    }
                    pred_xs.push(xi);
                    pred_ys.push(yi);
                    pred_mapped.push(isMapped ? (yi <= 25 ? yi : 25 + (yi - 25) * 5) : yi);
                }

                let modelLabel = useLog ? 'Log-fit' : 'Linear';
                let predTrace = {
                    x: pred_xs,
                    y: pred_mapped,
                    name: `${d.label} Pred (${modelLabel})`,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: d.color, width: 2, dash: 'dash' },
                    marker: { color: d.color, size: 5, symbol: 'circle-open' },
                    yaxis: d.axis
                };
                if (isMapped) {
                    predTrace.customdata = pred_ys;
                    predTrace.hovertemplate = '%{customdata:.2f} dB<extra></extra>';
                }
                plotData.push(predTrace);
            }
        }
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

    const showEpochs = document.getElementById('toggle-epochs')?.checked;
    if (showEpochs && filteredData.length > 0) {
        let shapes = [];
        let annotations = [];
        
        let startIter = filteredData[0].iteration;
        let startEpoch = filteredData[0].epoch || 1;
        
        for (let i = 1; i <= filteredData.length; i++) {
            let currentEpoch = i < filteredData.length ? filteredData[i].epoch : null;
            let currentIter = i < filteredData.length ? filteredData[i].iteration : filteredData[filteredData.length - 1].iteration;
            
            // if we hit a new epoch or end of data
            if (currentEpoch !== startEpoch || i === filteredData.length) {
                let isEven = startEpoch % 2 === 0;
                shapes.push({
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0: startIter,
                    y0: 0,
                    x1: currentIter,
                    y1: 1,
                    fillcolor: isEven ? 'rgba(255,255,255,0.015)' : 'rgba(255,255,255,0.05)',
                    line: { width: 0 },
                    layer: 'below'
                });
                
                // only add annotation if the band is wide enough
                if (currentIter - startIter > 1000) {
                    annotations.push({
                        x: (startIter + currentIter) / 2,
                        y: 1, // At the very top
                        xref: 'x',
                        yref: 'paper',
                        text: `Ep ${startEpoch}`,
                        showarrow: false,
                        font: { size: 10, color: 'rgba(255,255,255,0.3)' },
                        yanchor: 'bottom'
                    });
                }
                
                startIter = currentIter;
                startEpoch = currentEpoch;
            }
        }
        
        layout.shapes = shapes;
        layout.annotations = annotations;
        // make top margin slightly larger to accommodate epoch labels
        layout.margin.t = 20;
    }

    if (currentTab === 'losses') {
        layout.yaxis2 = {
            overlaying: 'y',
            side: 'right',
            showgrid: false,
            zeroline: false
        };
    } else if (currentTab === 'quality') {
        if (showRoC) {
            layout.yaxis2 = {
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                zeroline: true,
                zerolinecolor: 'rgba(255,45,120,0.3)',
                showticklabels: true
            };
        }
    }

    if (layout.yaxis2) {
        let min1 = Infinity, max1 = -Infinity;
        let min2 = Infinity, max2 = -Infinity;
        let hasY1 = false, hasY2 = false;

        plotData.forEach(p => {
            let yArr = p.y.filter(v => v != null);
            if (yArr.length === 0) return;
            let pmin = Math.min(...yArr);
            let pmax = Math.max(...yArr);
            if (p.yaxis === 'y' || p.yaxis === undefined) {
                min1 = Math.min(min1, pmin);
                max1 = Math.max(max1, pmax);
                hasY1 = true;
            } else if (p.yaxis === 'y2') {
                min2 = Math.min(min2, pmin);
                max2 = Math.max(max2, pmax);
                hasY2 = true;
            }
        });

        if (hasY1 && hasY2) {
            if (max1 === min1) { max1 += 1; min1 -= 1; }
            if (max2 === min2) { max2 += 1; min2 -= 1; }

            let pad1 = (max1 - min1) * 0.05; min1 -= pad1; max1 += pad1;
            let pad2 = (max2 - min2) * 0.05; min2 -= pad2; max2 += pad2;

            if (max1 > 0 && max2 > 0) {
                let f1 = max1 / (max1 - min1);
                let f2 = max2 / (max2 - min2);
                let f = Math.min(f1, f2);
                
                if (f > 0) {
                    if (f1 > f) min1 = max1 * (1 - 1/f);
                    else if (f2 > f) min2 = max2 * (1 - 1/f);
                }
            }

            layout.yaxis.range = [min1, max1];
            layout.yaxis2.range = [min2, max2];
            layout.yaxis.zeroline = true;
            layout.yaxis2.zeroline = true;
        }
    }

    if (currentTab === 'quality') {
        let tvals = [];
        let ttext = [];
        for(let i=-200; i<0; i+=20) { tvals.push(i); ttext.push(i.toString()); }
        for(let i=0; i<=25; i+=5) { tvals.push(i); ttext.push(i.toString()); }
        for(let i=26; i<=40; i+=1) { tvals.push(25 + (i - 25) * 5); ttext.push(i.toString()); }
        for(let i=42; i<=100; i+=2) { tvals.push(25 + (i - 25) * 5); ttext.push(i.toString()); }
        layout.yaxis.tickvals = tvals;
        layout.yaxis.ticktext = ttext;
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

