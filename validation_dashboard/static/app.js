let state = {
    data: [],
    filteredData: [],
    currentImg: null,
    scale: 1,
    translateX: 0,
    translateY: 0,
    isDragging: false,
    startX: 0,
    startY: 0,
    showDiff: false,
    showMask: false,
    showHeatmap: false,
    showGT: false,
    images: {
        gt: new Image(),
        m1: new Image(),
        m2: new Image(),
        m3: new Image(),
        m4: new Image(),
        m5: new Image(),
        mask: new Image()
    }
};

const elements = {
    searchInput: document.getElementById('searchInput'),
    filterBestModel: document.getElementById('filterBestModel'),
    sortSelect: document.getElementById('sortSelect'),
    sizeSelect: document.getElementById('sizeSelect'),
    statsDisplay: document.getElementById('statsDisplay'),
    toggleDiffBtn: document.getElementById('toggleDiffBtn'),
    resetZoomBtn: document.getElementById('resetZoomBtn'),
    imageList: document.getElementById('imageList'),
    
    modelSelects: [
        document.getElementById('model1Select'),
        document.getElementById('model2Select'),
        document.getElementById('model3Select'),
        document.getElementById('model4Select'),
        document.getElementById('model5Select')
    ],
    
    headers: [null, ...[1,2,3,4,5].map(i => document.getElementById(`header${i}`))],
    badges: [null, ...[1,2,3,4,5].map(i => document.getElementById(`badge${i}`))],
    panels: [document.getElementById('gtPanel'), ...[1,2,3,4,5].map(i => document.getElementById(`panel${i}`))],
    canvases: [document.getElementById('canvasGT'), ...[1,2,3,4,5].map(i => document.getElementById(`canvas${i}`))],
    hiddenCanvases: [document.getElementById('hiddenGT'), ...[1,2,3,4,5].map(i => document.getElementById(`hidden${i}`))],
    containers: [document.getElementById('containerGT'), ...[1,2,3,4,5].map(i => document.getElementById(`container${i}`))],
    
    hiddenMask: document.getElementById('hiddenMask'),
    voteComment: document.getElementById('voteComment'),
    saveVoteBtn: document.getElementById('saveVoteBtn'),
    voteButtons: document.querySelectorAll('.vote-btn')
};

const contexts = elements.canvases.map(c => c.getContext('2d', { willReadFrequently: true }));
const hiddenContexts = elements.hiddenCanvases.map(c => c.getContext('2d', { willReadFrequently: true }));
const maskCtx = elements.hiddenMask.getContext('2d', { willReadFrequently: true });

async function fetchFolders() {
    try {
        const response = await fetch('/api/folders');
        const folders = await response.json();
        elements.modelSelects.forEach((sel, i) => {
            sel.innerHTML = `<option value="">-- None --</option>`;
            folders.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f; opt.textContent = f;
                sel.appendChild(opt);
            });
            if (folders[i]) sel.value = folders[i];
        });
        fetchData();
    } catch (e) { console.error(e); }
}

async function fetchData() {
    const size = elements.sizeSelect.value;
    const query = new URLSearchParams({ size });
    elements.modelSelects.forEach((s, i) => { if (s.value) query.append(`model${i+1}`, s.value); });
    
    elements.statsDisplay.textContent = 'Loading...';
    try {
        const res = await fetch(`/api/data?${query.toString()}`);
        state.data = await res.json();
        applyFilters();
    } catch (e) { elements.statsDisplay.textContent = "Error."; }
}

function applyFilters() {
    const search = elements.searchInput.value.toLowerCase();
    const bestModel = elements.filterBestModel.value;
    const sort = elements.sortSelect.value;
    
    let filtered = state.data.filter(item => {
        if (search && !item.id.toLowerCase().includes(search)) return false;
        
        if (bestModel !== 'all') {
            const idx = parseInt(bestModel);
            const psnrs = [item.psnr_1, item.psnr_2, item.psnr_3, item.psnr_4, item.psnr_5].filter(p => p !== undefined);
            const max = Math.max(...psnrs);
            if (item[`psnr_${idx}`] !== max) return false;
        }
        return true;
    });
    
    filtered.sort((a, b) => {
        if (sort === 'id_asc') return a.id.localeCompare(b.id);
        if (sort === 'best_psnr_desc') {
            const getBest = x => Math.max(...[x.psnr_1, x.psnr_2, x.psnr_3, x.psnr_4, x.psnr_5].filter(p => p !== undefined));
            return getBest(b) - getBest(a);
        }
        if (sort.startsWith('psnr')) {
            const i = sort.split('_')[0].replace('psnr', '');
            return b[`psnr_${i}`] - a[`psnr_${i}`];
        }
        return 0;
    });
    
    state.filteredData = filtered;
    elements.statsDisplay.textContent = `${filtered.length} / ${state.data.length}`;
    renderGallery();
    
    if (filtered.length > 0 && (!state.currentImg || !filtered.find(d => d.id === state.currentImg.id))) {
        selectImage(filtered[0]);
    }
}

function renderGallery() {
    elements.imageList.innerHTML = '';
    const frag = document.createDocumentFragment();
    state.filteredData.forEach(item => {
        const div = document.createElement('div');
        div.className = `gallery-item ${state.currentImg?.id === item.id ? 'active' : ''}`;
        
        // Find best PSNR for thumb info
        const psnrs = [item.psnr_1, item.psnr_2, item.psnr_3, item.psnr_4, item.psnr_5].filter(p => p !== undefined);
        const best = Math.max(...psnrs);

        div.innerHTML = `
            <div class="thumb"><img src="${item.f1_fake}" loading="lazy"></div>
            <div class="info">
                <span class="id">${item.id}</span>
                <span class="psnr">Best: ${best.toFixed(2)}</span>
            </div>
        `;
        div.onclick = () => selectImage(item);
        frag.appendChild(div);
    });
    elements.imageList.appendChild(frag);
    
    // Scroll active item into view
    const active = elements.imageList.querySelector('.gallery-item.active');
    if (active) active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
}

function selectImage(item) {
    state.currentImg = item;
    renderGallery();
    
    let max = -1, bestIdx = -1;
    for (let i = 1; i <= 5; i++) {
        const p = item[`psnr_${i}`];
        if (p > max) { max = p; bestIdx = i; }
    }
    
    elements.modelSelects.forEach((sel, i) => {
        const idx = i + 1;
        const panel = elements.panels[idx], header = elements.headers[idx], badge = elements.badges[idx];
        header.textContent = sel.value || `Model ${idx}`;
        badge.textContent = `PSNR: ${item[`psnr_${idx}`]?.toFixed(2) || '--'}`;
        panel.classList.toggle('best-model', idx === bestIdx && max > 0);
        if (idx === bestIdx && max > 0) badge.textContent = `🏆 BEST: ${max.toFixed(2)}`;
        panel.style.display = sel.value ? 'flex' : 'none';
    });
    
    elements.voteComment.value = item.comment || '';
    loadImages(item);
}

function loadImages(item) {
    let toLoad = 1; // GT
    elements.modelSelects.forEach(s => { if (s.value) toLoad++; });
    let loaded = 0;
    const onDone = () => { if (++loaded === toLoad) { drawImages(); if (state.scale === 1) centerImages(); } };
    
    state.images.gt.onload = onDone;
    state.images.gt.src = item.gt;
    for (let i = 1; i <= 5; i++) {
        if (elements.modelSelects[i-1].value) {
            state.images[`m${i}`].onload = onDone;
            state.images[`m${i}`].src = item[`f${i}_fake`];
        }
    }
    // Load mask separately
    state.images.mask.onload = drawImages;
    const m1 = elements.modelSelects[0].value || 'none';
    const sz = elements.sizeSelect.value;
    state.images.mask.src = `/api/mask_only/${m1}/${sz}/${item.id}`;
}

function drawImages() {
    if (!state.currentImg) return;
    const gtImg = state.images.gt;
    if (!gtImg.complete || gtImg.naturalWidth === 0) return;

    const w = gtImg.width, h = gtImg.height;
    
    elements.canvases.forEach(c => { c.width = w; c.height = h; });
    elements.hiddenCanvases.forEach(c => { c.width = w; c.height = h; });
    elements.hiddenMask.width = w; elements.hiddenMask.height = h;
    
    hiddenContexts[0].drawImage(gtImg, 0, 0);
    contexts[0].drawImage(gtImg, 0, 0);
    
    const maskImg = state.images.mask;
    const isMaskReady = maskImg.complete && maskImg.naturalWidth > 0;
    if (isMaskReady) maskCtx.drawImage(maskImg, 0, 0, w, h);

    for (let i = 1; i <= 5; i++) {
        if (!elements.modelSelects[i-1].value) continue;
        const ctx = contexts[i], hCtx = hiddenContexts[i], img = state.images[`m${i}`];
        if (!img.complete || img.naturalWidth === 0) continue;
        
        hCtx.drawImage(img, 0, 0);
        
        if (state.showDiff) {
            const gt = hiddenContexts[0].getImageData(0,0,w,h), m = hCtx.getImageData(0,0,w,h), d = ctx.createImageData(w,h);
            for (let j=0; j<gt.data.length; j+=4) {
                d.data[j] = Math.abs(gt.data[j]-m.data[j])*2;
                d.data[j+1] = Math.abs(gt.data[j+1]-m.data[j+1])*2;
                d.data[j+2] = Math.abs(gt.data[j+2]-m.data[j+2])*2;
                d.data[j+3] = 255;
            }
            ctx.putImageData(d, 0, 0);
        } else if (state.showHeatmap && isMaskReady) {
            const gt = hiddenContexts[0].getImageData(0,0,w,h), m = hCtx.getImageData(0,0,w,h), d = ctx.createImageData(w,h);
            const mask = maskCtx.getImageData(0,0,w,h).data;
            for (let j=0; j<gt.data.length; j+=4) {
                const isMask = mask[j] === 255;
                if (isMask) {
                    const err = (Math.abs(gt.data[j]-m.data[j]) + Math.abs(gt.data[j+1]-m.data[j+1]) + Math.abs(gt.data[j+2]-m.data[j+2]))/3;
                    const r = Math.min(err/50, 1.0);
                    d.data[j] = r*255; d.data[j+1] = (1-r)*255; d.data[j+2] = 0; d.data[j+3] = 255;
                } else {
                    d.data[j] = m.data[j]*0.3; d.data[j+1] = m.data[j+1]*0.3; d.data[j+2] = m.data[j+2]*0.3; d.data[j+3] = 255;
                }
            }
            ctx.putImageData(d, 0, 0);
        } else {
            ctx.drawImage(img, 0, 0);
        }
        
        if (state.showMask && isMaskReady) {
            ctx.globalCompositeOperation = 'lighten';
            ctx.drawImage(maskImg, 0, 0, w, h);
            ctx.globalCompositeOperation = 'source-over';
        }
    }
    
    if (state.showGT) {
        elements.canvases.forEach((c, idx) => { if(idx > 0) contexts[idx].drawImage(gtImg, 0, 0); });
    }
    updateTransforms();
}

function centerImages() {
    const c = elements.containers[0], cw = c.clientWidth, ch = c.clientHeight, iw = state.images.gt.width, ih = state.images.gt.height;
    state.scale = Math.min((cw-20)/iw, (ch-20)/ih, 5);
    state.translateX = (cw - iw*state.scale)/2; state.translateY = (ch - ih*state.scale)/2;
    updateTransforms();
}

function updateTransforms() {
    const t = `translate(${state.translateX}px, ${state.translateY}px) scale(${state.scale})`;
    elements.canvases.forEach(c => c.style.transform = t);
}

// Controls
elements.containers.forEach(c => {
    c.onwheel = e => {
        e.preventDefault();
        const r = c.getBoundingClientRect(), mx = e.clientX-r.left, my = e.clientY-r.top;
        const z = Math.exp((e.deltaY<0?1:-1)*0.1), ns = Math.max(0.1, Math.min(state.scale*z, 50));
        state.translateX = mx - (mx-state.translateX)*(ns/state.scale);
        state.translateY = my - (my-state.translateY)*(ns/state.scale);
        state.scale = ns; updateTransforms();
    };
    c.onmousedown = e => { state.isDragging = true; state.startX = e.clientX-state.translateX; state.startY = e.clientY-state.translateY; };
});

window.onmousemove = e => { if (state.isDragging) { state.translateX = e.clientX-state.startX; state.translateY = e.clientY-state.startY; updateTransforms(); } };
window.onmouseup = () => state.isDragging = false;

elements.searchInput.oninput = applyFilters;
elements.filterBestModel.onchange = applyFilters;
elements.sortSelect.onchange = applyFilters;
elements.sizeSelect.onchange = fetchData;
elements.modelSelects.forEach(s => s.onchange = fetchData);
elements.resetZoomBtn.onclick = centerImages;
elements.toggleDiffBtn.onclick = () => { state.showDiff = !state.showDiff; elements.toggleDiffBtn.classList.toggle('active', state.showDiff); drawImages(); };

// Shortcuts
window.onkeydown = e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const idx = state.filteredData.findIndex(d => d.id === state.currentImg?.id);
    if (e.key === 'ArrowRight' && idx < state.filteredData.length-1) selectImage(state.filteredData[idx+1]);
    if (e.key === 'ArrowLeft' && idx > 0) selectImage(state.filteredData[idx-1]);
    if (e.key.toLowerCase() === 'm') { state.showMask = true; drawImages(); }
    if (e.key.toLowerCase() === 'n') { state.showHeatmap = true; drawImages(); }
    if (e.key.toLowerCase() === 'b') { state.showGT = true; drawImages(); }
};

window.onkeyup = e => {
    if (e.key.toLowerCase() === 'm') { state.showMask = false; drawImages(); }
    if (e.key.toLowerCase() === 'n') { state.showHeatmap = false; drawImages(); }
    if (e.key.toLowerCase() === 'b') { state.showGT = false; drawImages(); }
};

fetchFolders();
