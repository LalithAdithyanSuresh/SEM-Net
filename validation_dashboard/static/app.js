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
    numModels: 5,
    images: {
        gt: new Image(),
        m1: new Image(),
        m2: new Image(),
        m3: new Image(),
        m4: new Image(),
        m5: new Image()
    }
};

const elements = {
    searchInput: document.getElementById('searchInput'),
    imageDropdown: document.getElementById('imageDropdown'),
    sortSelect: document.getElementById('sortSelect'),
    sizeSelect: document.getElementById('sizeSelect'),
    statsDisplay: document.getElementById('statsDisplay'),
    toggleDiffBtn: document.getElementById('toggleDiffBtn'),
    resetZoomBtn: document.getElementById('resetZoomBtn'),
    
    modelSelects: [
        document.getElementById('model1Select'),
        document.getElementById('model2Select'),
        document.getElementById('model3Select'),
        document.getElementById('model4Select'),
        document.getElementById('model5Select')
    ],
    
    headers: [
        null, // index 0 is GT
        document.getElementById('header1'),
        document.getElementById('header2'),
        document.getElementById('header3'),
        document.getElementById('header4'),
        document.getElementById('header5')
    ],
    
    badges: [
        null,
        document.getElementById('badge1'),
        document.getElementById('badge2'),
        document.getElementById('badge3'),
        document.getElementById('badge4'),
        document.getElementById('badge5')
    ],
    
    panels: [
        document.getElementById('gtPanel'),
        document.getElementById('panel1'),
        document.getElementById('panel2'),
        document.getElementById('panel3'),
        document.getElementById('panel4'),
        document.getElementById('panel5')
    ],
    
    canvases: [
        document.getElementById('canvasGT'),
        document.getElementById('canvas1'),
        document.getElementById('canvas2'),
        document.getElementById('canvas3'),
        document.getElementById('canvas4'),
        document.getElementById('canvas5')
    ],
    
    hiddenCanvases: [
        document.getElementById('hiddenGT'),
        document.getElementById('hidden1'),
        document.getElementById('hidden2'),
        document.getElementById('hidden3'),
        document.getElementById('hidden4'),
        document.getElementById('hidden5')
    ],
    
    containers: [
        document.getElementById('containerGT'),
        document.getElementById('container1'),
        document.getElementById('container2'),
        document.getElementById('container3'),
        document.getElementById('container4'),
        document.getElementById('container5')
    ]
};

const contexts = elements.canvases.map(c => c.getContext('2d', { willReadFrequently: true }));
const hiddenContexts = elements.hiddenCanvases.map(c => c.getContext('2d', { willReadFrequently: true }));

async function fetchFolders() {
    try {
        const response = await fetch('/api/folders');
        const folders = await response.json();
        
        elements.modelSelects.forEach((sel, i) => {
            sel.innerHTML = `<option value="">-- No Model --</option>`;
            folders.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f; opt.textContent = f;
                sel.appendChild(opt);
            });
            // Default selection
            if (folders[i]) sel.value = folders[i];
        });
        
        fetchData();
    } catch (e) {
        console.error("Failed to load folders", e);
    }
}

async function fetchData() {
    const size = elements.sizeSelect.value;
    const queryParams = new URLSearchParams({ size });
    elements.modelSelects.forEach((sel, i) => {
        if (sel.value) queryParams.append(`model${i+1}`, sel.value);
    });
    
    elements.statsDisplay.textContent = 'Loading...';
    
    try {
        const response = await fetch(`/api/data?${queryParams.toString()}`);
        state.data = await response.json();
        
        applyFilters();
    } catch (e) {
        console.error("Failed to fetch data", e);
        elements.statsDisplay.textContent = "Error.";
    }
}

function applyFilters() {
    const search = elements.searchInput.value.toLowerCase();
    const sort = elements.sortSelect.value;
    
    let filtered = state.data.filter(item => {
        if (search && !item.id.toLowerCase().includes(search)) return false;
        return true;
    });
    
    filtered.sort((a, b) => {
        if (sort === 'id_asc') return a.id.localeCompare(b.id);
        if (sort === 'id_desc') return b.id.localeCompare(a.id);
        
        if (sort === 'best_psnr_desc') {
            const getBest = (x) => Math.max(...[x.psnr_1, x.psnr_2, x.psnr_3, x.psnr_4, x.psnr_5].filter(v => v !== undefined));
            return getBest(b) - getBest(a);
        }
        
        if (sort.startsWith('psnr')) {
            const idx = sort.split('_')[0].replace('psnr', '');
            return b[`psnr_${idx}`] - a[`psnr_${idx}`];
        }
        
        return 0;
    });
    
    state.filteredData = filtered;
    elements.statsDisplay.textContent = `${filtered.length} / ${state.data.length}`;
    
    updateImageDropdown();
    
    if (state.filteredData.length > 0) {
        // Only auto-select if no current image or current image is not in filtered data
        if (!state.currentImg || !state.filteredData.find(d => d.id === state.currentImg.id)) {
            selectImage(state.filteredData[0]);
        }
    }
}

function updateImageDropdown() {
    const val = elements.imageDropdown.value;
    elements.imageDropdown.innerHTML = `<option value="">Select Image (${state.filteredData.length})...</option>`;
    state.filteredData.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.id;
        opt.textContent = `${item.id} (Best: ${Math.max(...[item.psnr_1, item.psnr_2, item.psnr_3, item.psnr_4, item.psnr_5].filter(v => v !== undefined)).toFixed(2)})`;
        elements.imageDropdown.appendChild(opt);
    });
    elements.imageDropdown.value = state.currentImg ? state.currentImg.id : '';
}

function selectImage(item) {
    if (!item) return;
    state.currentImg = item;
    elements.imageDropdown.value = item.id;
    
    // Determine Best Model
    let maxPsnr = -1;
    let bestIdx = -1;
    for (let i = 1; i <= 5; i++) {
        const p = item[`psnr_${i}`];
        if (p > maxPsnr) {
            maxPsnr = p;
            bestIdx = i;
        }
    }
    
    // Update UI headers and badges
    elements.modelSelects.forEach((sel, i) => {
        const idx = i + 1;
        const panel = elements.panels[idx];
        const header = elements.headers[idx];
        const badge = elements.badges[idx];
        
        header.textContent = sel.value || `Model ${idx}`;
        badge.textContent = `PSNR: ${item[`psnr_${idx}`]?.toFixed(2) || '--'}`;
        
        panel.classList.remove('best-model');
        if (idx === bestIdx && maxPsnr > 0) {
            panel.classList.add('best-model');
            badge.textContent = `🏆 BEST: ${maxPsnr.toFixed(2)}`;
        }
        
        // Hide panel if no model selected
        panel.style.display = sel.value ? 'flex' : 'none';
    });
    
    loadImages(item);
}

function loadImages(item) {
    let toLoad = 1; // GT
    elements.modelSelects.forEach(sel => { if (sel.value) toLoad++; });
    
    let loaded = 0;
    const onLoaded = () => {
        loaded++;
        if (loaded === toLoad) {
            drawImages();
            if (state.scale === 1) centerImages();
        }
    };
    
    state.images.gt.onload = onLoaded;
    state.images.gt.src = item.gt;
    
    for (let i = 1; i <= 5; i++) {
        const sel = elements.modelSelects[i-1];
        if (sel.value) {
            state.images[`m${i}`].onload = onLoaded;
            state.images[`m${i}`].src = item[`f${i}_fake`];
        } else {
            state.images[`m${i}`].src = '';
        }
    }
}

function drawImages() {
    if (!state.currentImg) return;
    
    const w = state.images.gt.width;
    const h = state.images.gt.height;
    if (w === 0) return;
    
    // Resize all canvases
    elements.canvases.forEach(c => { c.width = w; c.height = h; });
    elements.hiddenCanvases.forEach(c => { c.width = w; c.height = h; });
    
    // Draw base images to hidden canvases
    hiddenContexts[0].drawImage(state.images.gt, 0, 0);
    for (let i = 1; i <= 5; i++) {
        if (elements.modelSelects[i-1].value) {
            hiddenContexts[i].drawImage(state.images[`m${i}`], 0, 0);
        }
    }
    
    // Draw to visible canvases
    contexts[0].drawImage(state.images.gt, 0, 0);
    
    for (let i = 1; i <= 5; i++) {
        if (!elements.modelSelects[i-1].value) continue;
        
        const ctx = contexts[i];
        if (state.showDiff) {
            const gtData = hiddenContexts[0].getImageData(0, 0, w, h);
            const mData = hiddenContexts[i].getImageData(0, 0, w, h);
            const diffData = ctx.createImageData(w, h);
            
            for (let j = 0; j < gtData.data.length; j += 4) {
                diffData.data[j] = Math.abs(gtData.data[j] - mData.data[j]) * 2; // Boost diff
                diffData.data[j+1] = Math.abs(gtData.data[j+1] - mData.data[j+1]) * 2;
                diffData.data[j+2] = Math.abs(gtData.data[j+2] - mData.data[j+2]) * 2;
                diffData.data[j+3] = 255;
            }
            ctx.putImageData(diffData, 0, 0);
        } else {
            ctx.drawImage(state.images[`m${i}`], 0, 0);
        }
    }
    
    updateTransforms();
}

function centerImages() {
    const container = elements.containers[0];
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const iw = state.images.gt.width;
    const ih = state.images.gt.height;
    
    const scale = Math.min((cw - 20) / iw, (ch - 20) / ih, 5);
    state.scale = scale;
    state.translateX = (cw - iw * scale) / 2;
    state.translateY = (ch - ih * scale) / 2;
    
    updateTransforms();
}

function updateTransforms() {
    const transform = `translate(${state.translateX}px, ${state.translateY}px) scale(${state.scale})`;
    elements.canvases.forEach(c => {
        c.style.transform = transform;
    });
}

// Global Zoom/Pan sync
elements.containers.forEach(container => {
    container.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = container.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const zoom = Math.exp((e.deltaY < 0 ? 1 : -1) * 0.1);
        const newScale = Math.max(0.1, Math.min(state.scale * zoom, 50));
        
        state.translateX = mouseX - (mouseX - state.translateX) * (newScale / state.scale);
        state.translateY = mouseY - (mouseY - state.translateY) * (newScale / state.scale);
        state.scale = newScale;
        updateTransforms();
    }, { passive: false });
    
    container.addEventListener('mousedown', (e) => {
        state.isDragging = true;
        state.startX = e.clientX - state.translateX;
        state.startY = e.clientY - state.translateY;
    });
});

window.addEventListener('mousemove', (e) => {
    if (!state.isDragging) return;
    state.translateX = e.clientX - state.startX;
    state.translateY = e.clientY - state.startY;
    updateTransforms();
});

window.addEventListener('mouseup', () => state.isDragging = false);

// UI Listeners
elements.searchInput.addEventListener('input', applyFilters);
elements.imageDropdown.addEventListener('change', () => {
    const item = state.filteredData.find(d => d.id === elements.imageDropdown.value);
    if (item) selectImage(item);
});
elements.sortSelect.addEventListener('change', applyFilters);
elements.sizeSelect.addEventListener('change', fetchData);
elements.modelSelects.forEach(sel => sel.addEventListener('change', fetchData));
elements.resetZoomBtn.addEventListener('click', centerImages);
elements.toggleDiffBtn.addEventListener('click', () => {
    state.showDiff = !state.showDiff;
    elements.toggleDiffBtn.classList.toggle('active', state.showDiff);
    drawImages();
});

// Shortcuts
window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    const idx = state.filteredData.findIndex(d => d.id === state.currentImg?.id);
    if (e.key === 'ArrowRight' && idx < state.filteredData.length - 1) selectImage(state.filteredData[idx + 1]);
    if (e.key === 'ArrowLeft' && idx > 0) selectImage(state.filteredData[idx - 1]);
});

window.addEventListener('resize', centerImages);

fetchFolders();
