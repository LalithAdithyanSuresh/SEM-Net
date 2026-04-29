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
        m1: new Image(),
        gt: new Image(),
        m2: new Image()
    }
};

const elements = {
    imageList: document.getElementById('imageList'),
    searchInput: document.getElementById('searchInput'),
    sortSelect: document.getElementById('sortSelect'),
    filterPsnr1: document.getElementById('filterPsnr1'),
    filterPsnr2: document.getElementById('filterPsnr2'),
    filterStatus: document.getElementById('filterStatus'),
    statsDisplay: document.getElementById('statsDisplay'),
    
    title: document.getElementById('currentImageTitle'),
    maskPercentageBadge: document.getElementById('maskPercentageBadge'),
    m1PsnrBadge: document.getElementById('m1PsnrBadge'),
    m2PsnrBadge: document.getElementById('m2PsnrBadge'),
    
    viewScoredBtn: document.getElementById('viewScoredBtn'),
    toggleGtBtn: document.getElementById('toggleGtBtn'),
    toggleDiffBtn: document.getElementById('toggleDiffBtn'),
    resetZoomBtn: document.getElementById('resetZoomBtn'),
    sizeSelect: document.getElementById('sizeSelect'),
    model1Select: document.getElementById('model1Select'),
    model2Select: document.getElementById('model2Select'),
    
    panelHeader1: document.getElementById('panelHeader1'),
    panelHeader2: document.getElementById('panelHeader2'),
    voteText1: document.getElementById('voteText1'),
    voteText2: document.getElementById('voteText2'),
    gridHeader1: document.getElementById('gridHeader1'),
    gridHeader2: document.getElementById('gridHeader2'),
    
    gtPanel: document.getElementById('gtPanel'),
    
    canvas1: document.getElementById('canvas1'),
    canvasGT: document.getElementById('canvasGT'),
    canvas2: document.getElementById('canvas2'),
    
    hidden1: document.getElementById('hiddenCanvas1'),
    hidden2: document.getElementById('hiddenCanvas2'),
    
    grid1Img: document.getElementById('grid1Img'),
    grid2Img: document.getElementById('grid2Img'),
    
    voteM1Btn: document.getElementById('voteM1Btn'),
    voteTieBtn: document.getElementById('voteTieBtn'),
    voteM2Btn: document.getElementById('voteM2Btn'),
    voteComment: document.getElementById('voteComment'),
    saveVoteBtn: document.getElementById('saveVoteBtn'),
    
    containers: [
        document.getElementById('container1'),
        document.getElementById('containerGT'),
        document.getElementById('container2')
    ]
};

const contexts = {
    c1: elements.canvas1.getContext('2d', { willReadFrequently: true }),
    cGT: elements.canvasGT.getContext('2d'),
    c2: elements.canvas2.getContext('2d', { willReadFrequently: true }),
    h1: elements.hidden1.getContext('2d', { willReadFrequently: true }),
    h2: elements.hidden2.getContext('2d', { willReadFrequently: true })
};

async function fetchFolders() {
    try {
        const response = await fetch('/api/folders');
        const folders = await response.json();
        
        elements.model1Select.innerHTML = '';
        elements.model2Select.innerHTML = '';
        
        folders.forEach(f => {
            const opt1 = document.createElement('option');
            opt1.value = f; opt1.textContent = f;
            elements.model1Select.appendChild(opt1);
            
            const opt2 = document.createElement('option');
            opt2.value = f; opt2.textContent = f;
            elements.model2Select.appendChild(opt2);
        });
        
        if (folders.includes('evaluation_results_standard_uniform')) {
            elements.model1Select.value = 'evaluation_results_standard_uniform';
        } else if (folders.length > 0) {
            elements.model1Select.value = folders[0];
        }
        
        if (folders.includes('deterministic_strided')) {
            elements.model2Select.value = 'deterministic_strided';
        } else if (folders.length > 1) {
            elements.model2Select.value = folders[1];
        } else if (folders.length > 0) {
            elements.model2Select.value = folders[0];
        }
        
        updateModelLabels();
        fetchData();
    } catch (e) {
        console.error("Failed to load folders", e);
    }
}

function updateModelLabels() {
    const m1 = elements.model1Select.value;
    const m2 = elements.model2Select.value;
    
    const l1 = m1 === 'evaluation_results_standard_uniform' ? 'ONLY VA MAMBA' : m1;
    const l2 = m2 === 'deterministic_strided' ? 'VA + DA MAMBA' : m2;
    
    elements.panelHeader1.textContent = l1;
    elements.panelHeader2.textContent = l2;
    elements.voteText1.textContent = l1;
    elements.voteText2.textContent = l2;
    elements.gridHeader1.textContent = `${l1} - 5 Image Grid`;
    elements.gridHeader2.textContent = `${l2} - 5 Image Grid`;
}

async function fetchData() {
    const size = elements.sizeSelect.value;
    const m1 = elements.model1Select.value;
    const m2 = elements.model2Select.value;
    
    if (!m1 || !m2) return;
    
    elements.statsDisplay.textContent = 'Loading...';
    
    try {
        const response = await fetch(`/api/data?size=${size}&model1=${m1}&model2=${m2}`);
        state.data = await response.json();
        
        // Calculate diff PSNR for sorting
        state.data.forEach(item => {
            item.diff_psnr = item.psnr1 - item.psnr2; // or abs
        });
        
        applyFilters();
    } catch (e) {
        console.error("Failed to fetch data", e);
        elements.statsDisplay.textContent = "Error loading data.";
    }
}

function applyFilters() {
    const search = elements.searchInput.value.toLowerCase();
    const minPsnr1 = parseFloat(elements.filterPsnr1.value) || -Infinity;
    const minPsnr2 = parseFloat(elements.filterPsnr2.value) || -Infinity;
    const sort = elements.sortSelect.value;
    const status = elements.filterStatus.value;
    
    let filtered = state.data.filter(item => {
        if (search && !item.id.toLowerCase().includes(search)) return false;
        if (item.psnr_1 < minPsnr1) return false;
        if (item.psnr_2 < minPsnr2) return false;
        
        if (status === 'unscored' && item.winner) return false;
        if (status === 'scored' && !item.winner) return false;
        if (status === 'm1' && item.winner !== 'm1') return false;
        if (status === 'm2' && item.winner !== 'm2') return false;
        if (status === 'tie' && item.winner !== 'tie') return false;
        if (status === 'commented' && !item.comment) return false;
        
        return true;
    });
    
    filtered.sort((a, b) => {
        switch(sort) {
            case 'id_asc': return a.id.localeCompare(b.id);
            case 'id_desc': return b.id.localeCompare(a.id);
            case 'psnr1_asc': return a.psnr_1 - b.psnr_1;
            case 'psnr1_desc': return b.psnr_1 - a.psnr_1;
            case 'psnr2_asc': return a.psnr_2 - b.psnr_2;
            case 'psnr2_desc': return b.psnr_2 - a.psnr_2;
            case 'diff_psnr_desc': return Math.abs(b.psnr_1 - b.psnr_2) - Math.abs(a.psnr_1 - a.psnr_2);
            default: return 0;
        }
    });
    
    state.filteredData = filtered;
    elements.statsDisplay.textContent = `Showing ${filtered.length} / ${state.data.length} images`;
    
    renderList();
    
    if (state.filteredData.length > 0 && !state.currentImg) {
        selectImage(state.filteredData[0]);
    }
}

function renderList() {
    elements.imageList.innerHTML = '';
    
    const fragment = document.createDocumentFragment();
    state.filteredData.forEach(item => {
        const div = document.createElement('div');
        div.className = 'list-item';
        if (state.currentImg && state.currentImg.id === item.id) {
            div.classList.add('active');
        }
        
        
        let indicator = '';
        if (item.winner === 'm1') indicator = '<span class="vote-indicator vote-m1">M1</span>';
        else if (item.winner === 'm2') indicator = '<span class="vote-indicator vote-m2">M2</span>';
        else if (item.winner === 'tie') indicator = '<span class="vote-indicator vote-tie">Tie</span>';

        div.innerHTML = `
            <div class="item-id">${item.id} ${indicator}</div>
            <div class="item-metrics">
                <span>M1: ${item.psnr_1.toFixed(2)}</span>
                <span>M2: ${item.psnr_2.toFixed(2)}</span>
            </div>
        `;
        
        div.addEventListener('click', () => selectImage(item));
        fragment.appendChild(div);
    });
    
    elements.imageList.appendChild(fragment);
}

function selectImage(item) {
    state.currentImg = item;
    
    // Update active class in list
    document.querySelectorAll('.list-item').forEach(el => {
        el.classList.remove('active');
        if (el.querySelector('.item-id').textContent === item.id) {
            el.classList.add('active');
        }
    });
    
    elements.title.textContent = `Image: ${item.id}`;
    elements.m1PsnrBadge.textContent = `M1 PSNR: ${item.psnr_1.toFixed(2)}`;
    elements.m2PsnrBadge.textContent = `M2 PSNR: ${item.psnr_2.toFixed(2)}`;
    
    elements.grid1Img.src = item.f1_grid || '';
    elements.grid2Img.src = item.f2_grid || '';
    
    // Setup voting UI
    elements.voteComment.value = item.comment || '';
    
    elements.voteM1Btn.classList.remove('selected');
    elements.voteTieBtn.classList.remove('selected');
    elements.voteM2Btn.classList.remove('selected');
    
    if (item.winner === 'm1') elements.voteM1Btn.classList.add('selected');
    else if (item.winner === 'tie') elements.voteTieBtn.classList.add('selected');
    else if (item.winner === 'm2') elements.voteM2Btn.classList.add('selected');
    
    loadCanvases(item);
}

function loadCanvases(item) {
    let loaded = 0;
    const checkLoaded = () => {
        loaded++;
        if (loaded === 3) {
            drawImages();
            if (state.scale === 1) centerImages();
        }
    };
    
    state.images.m1.onload = checkLoaded;
    state.images.gt.onload = checkLoaded;
    state.images.m2.onload = checkLoaded;
    
    state.images.m1.src = item.f1_fake;
    state.images.gt.src = item.gt;
    state.images.m2.src = item.f2_fake;
}

function drawImages() {
    if (!state.currentImg) return;
    
    const w = state.images.m1.width;
    const h = state.images.m1.height;
    
    [elements.canvas1, elements.canvasGT, elements.canvas2, elements.hidden1, elements.hidden2].forEach(c => {
        c.width = w;
        c.height = h;
    });
    
    contexts.h1.drawImage(state.images.m1, 0, 0);
    contexts.cGT.drawImage(state.images.gt, 0, 0);
    contexts.h2.drawImage(state.images.m2, 0, 0);
    
    if (state.showDiff) {
        // Calculate difference
        const imgData1 = contexts.h1.getImageData(0, 0, w, h);
        const imgData2 = contexts.h2.getImageData(0, 0, w, h);
        const diffData = contexts.c1.createImageData(w, h);
        
        for (let i = 0; i < imgData1.data.length; i += 4) {
            // Absolute difference for RGB
            diffData.data[i] = Math.abs(imgData1.data[i] - imgData2.data[i]);
            diffData.data[i+1] = Math.abs(imgData1.data[i+1] - imgData2.data[i+1]);
            diffData.data[i+2] = Math.abs(imgData1.data[i+2] - imgData2.data[i+2]);
            diffData.data[i+3] = 255; // Alpha
            
            // Boost visibility of diff
            // diffData.data[i] = Math.min(255, diffData.data[i] * 5);
            // diffData.data[i+1] = Math.min(255, diffData.data[i+1] * 5);
            // diffData.data[i+2] = Math.min(255, diffData.data[i+2] * 5);
        }
        
        contexts.c1.putImageData(diffData, 0, 0);
        contexts.c2.putImageData(diffData, 0, 0); // Show diff on both or just leave M2 as is? 
        // Let's show diff on Canvas 1, and keep M2 as M2
        contexts.c2.drawImage(state.images.m2, 0, 0);
        
        elements.toggleDiffBtn.textContent = "Showing Diff (M1) vs M2";
        elements.toggleDiffBtn.classList.add('active-toggle');
    } else {
        contexts.c1.drawImage(state.images.m1, 0, 0);
        contexts.c2.drawImage(state.images.m2, 0, 0);
        
        elements.toggleDiffBtn.textContent = "Toggle Pixel Diff (M1 vs M2)";
        elements.toggleDiffBtn.classList.remove('active-toggle');
    }
    
    // Calculate and display mask percentage
    if (elements.grid1Img.complete && elements.grid1Img.naturalWidth > 0) {
        contexts.h1.drawImage(elements.grid1Img, w, 0, w, h, 0, 0, w, h);
        const maskData = contexts.h1.getImageData(0, 0, w, h).data;
        let whiteCount = 0;
        for (let i = 0; i < maskData.length; i += 4) {
            if (maskData[i] === 255 && maskData[i+1] === 255 && maskData[i+2] === 255) {
                whiteCount++;
            }
        }
        const ratio = (whiteCount / (w * h)) * 100;
        elements.maskPercentageBadge.textContent = `Mask: ${ratio.toFixed(1)}%`;
    } else {
        elements.maskPercentageBadge.textContent = `Mask: --%`;
    }
    
    if (state.showMask && elements.grid1Img.complete) {
        // The masked image is the 2nd image in the 5-image grid
        [contexts.c1, contexts.cGT, contexts.c2].forEach(ctx => {
            ctx.drawImage(elements.grid1Img, w, 0, w, h, 0, 0, w, h);
        });
    } else if (state.showHeatmap && elements.grid1Img.complete) {
        contexts.h1.drawImage(elements.grid1Img, w, 0, w, h, 0, 0, w, h);
        const maskData = contexts.h1.getImageData(0, 0, w, h).data;
        const gtData = contexts.cGT.getImageData(0, 0, w, h).data;
        
        contexts.h1.drawImage(state.images.m1, 0, 0);
        contexts.h2.drawImage(state.images.m2, 0, 0);
        const m1Data = contexts.h1.getImageData(0, 0, w, h).data;
        const m2Data = contexts.h2.getImageData(0, 0, w, h).data;
        
        const heat1 = contexts.c1.createImageData(w, h);
        const heat2 = contexts.c2.createImageData(w, h);
        
        for (let i = 0; i < maskData.length; i += 4) {
            const isMaskWhite = (maskData[i] === 255 && maskData[i+1] === 255 && maskData[i+2] === 255);
            
            if (isMaskWhite) {
                const err1 = (Math.abs(m1Data[i] - gtData[i]) + Math.abs(m1Data[i+1] - gtData[i+1]) + Math.abs(m1Data[i+2] - gtData[i+2])) / 3;
                const err2 = (Math.abs(m2Data[i] - gtData[i]) + Math.abs(m2Data[i+1] - gtData[i+1]) + Math.abs(m2Data[i+2] - gtData[i+2])) / 3;
                
                const maxError = 50; // Calibrated for visible gradients
                
                let r1 = Math.min(err1 / maxError, 1.0);
                heat1.data[i] = Math.floor(r1 * 255);
                heat1.data[i+1] = Math.floor((1 - r1) * 255);
                heat1.data[i+2] = 0;
                heat1.data[i+3] = 255;
                
                let r2 = Math.min(err2 / maxError, 1.0);
                heat2.data[i] = Math.floor(r2 * 255);
                heat2.data[i+1] = Math.floor((1 - r2) * 255);
                heat2.data[i+2] = 0;
                heat2.data[i+3] = 255;
            } else {
                heat1.data[i] = m1Data[i] * 0.3;
                heat1.data[i+1] = m1Data[i+1] * 0.3;
                heat1.data[i+2] = m1Data[i+2] * 0.3;
                heat1.data[i+3] = 255;
                
                heat2.data[i] = m2Data[i] * 0.3;
                heat2.data[i+1] = m2Data[i+1] * 0.3;
                heat2.data[i+2] = m2Data[i+2] * 0.3;
                heat2.data[i+3] = 255;
            }
        }
        
        contexts.c1.putImageData(heat1, 0, 0);
        contexts.c2.putImageData(heat2, 0, 0);
    } else if (state.showGT) {
        contexts.c1.drawImage(state.images.gt, 0, 0);
        contexts.c2.drawImage(state.images.gt, 0, 0);
    }
    
    updateTransforms();
}

function centerImages() {
    if (!state.currentImg) return;
    const container = elements.containers[0];
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    
    const iw = state.images.m1.width;
    const ih = state.images.m1.height;
    
    // Fit to container with some padding
    const scaleX = (cw - 40) / iw;
    const scaleY = (ch - 40) / ih;
    state.scale = Math.min(scaleX, scaleY, 3); // Max scale 3 initially
    
    state.translateX = (cw - iw * state.scale) / 2;
    state.translateY = (ch - ih * state.scale) / 2;
    
    updateTransforms();
}

function updateTransforms() {
    const transform = `translate(${state.translateX}px, ${state.translateY}px) scale(${state.scale})`;
    elements.canvas1.style.transform = transform;
    elements.canvasGT.style.transform = transform;
    elements.canvas2.style.transform = transform;
}

// Event Listeners for sync pan/zoom
elements.containers.forEach(container => {
    container.addEventListener('wheel', (e) => {
        e.preventDefault();
        
        const rect = container.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const zoomIntensity = 0.1;
        const wheel = e.deltaY < 0 ? 1 : -1;
        
        const zoom = Math.exp(wheel * zoomIntensity);
        const newScale = Math.max(0.1, Math.min(state.scale * zoom, 20));
        
        if (newScale === state.scale) return;
        
        // Adjust translation to zoom toward mouse pointer
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
    
    window.addEventListener('mousemove', (e) => {
        if (!state.isDragging) return;
        state.translateX = e.clientX - state.startX;
        state.translateY = e.clientY - state.startY;
        updateTransforms();
    });
    
    window.addEventListener('mouseup', () => {
        state.isDragging = false;
    });
});

// UI Event Listeners
elements.searchInput.addEventListener('input', applyFilters);
elements.sortSelect.addEventListener('change', applyFilters);
elements.filterPsnr1.addEventListener('input', applyFilters);
elements.filterPsnr2.addEventListener('input', applyFilters);
elements.filterStatus.addEventListener('change', applyFilters);
elements.sizeSelect.addEventListener('change', fetchData);

elements.model1Select.addEventListener('change', () => {
    updateModelLabels();
    fetchData();
});

elements.model2Select.addEventListener('change', () => {
    updateModelLabels();
    fetchData();
});

elements.viewScoredBtn.addEventListener('click', () => {
    elements.filterStatus.value = 'scored';
    applyFilters();
});

elements.toggleDiffBtn.addEventListener('click', () => {
    state.showDiff = !state.showDiff;
    drawImages();
});

elements.resetZoomBtn.addEventListener('click', () => {
    centerImages();
});

window.addEventListener('resize', () => {
    if (state.scale === 1) centerImages();
});

// Voting Logic
async function saveVote(winner) {
    if (!state.currentImg) return;
    
    const image_id = state.currentImg.id;
    const size = elements.sizeSelect.value;
    const m1 = elements.model1Select.value;
    const m2 = elements.model2Select.value;
    const comment = elements.voteComment.value;
    
    // Update local state immediately for fast feedback
    if (winner !== undefined) {
        state.currentImg.winner = winner;
    } else {
        winner = state.currentImg.winner; // Just saving comment
    }
    
    state.currentImg.comment = comment;
    
    // Trigger visual update
    selectImage(state.currentImg);
    renderList();
    
    try {
        await fetch('/api/vote', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: image_id,
                size: size,
                model1: m1,
                model2: m2,
                winner: winner,
                comment: comment
            })
        });
    } catch (e) {
        console.error("Failed to save vote", e);
    }
}

async function syncVotes() {
    const size = elements.sizeSelect.value;
    const m1 = elements.model1Select.value;
    const m2 = elements.model2Select.value;
    if (!m1 || !m2 || !state.data || state.data.length === 0) return;
    
    try {
        const res = await fetch(`/api/votes_sync?size=${size}&model1=${m1}&model2=${m2}`);
        const dict = await res.json();
        
        let changed = false;
        state.data.forEach(item => {
            const voteData = dict[item.id] || {winner: null, comment: ''};
            if (item.winner !== voteData.winner || item.comment !== voteData.comment) {
                item.winner = voteData.winner;
                item.comment = voteData.comment;
                changed = true;
                
                // update current if looking at it
                if (state.currentImg && state.currentImg.id === item.id) {
                    elements.voteComment.value = item.comment;
                    elements.voteM1Btn.classList.remove('selected');
                    elements.voteTieBtn.classList.remove('selected');
                    elements.voteM2Btn.classList.remove('selected');
                    
                    if (item.winner === 'm1') elements.voteM1Btn.classList.add('selected');
                    else if (item.winner === 'tie') elements.voteTieBtn.classList.add('selected');
                    else if (item.winner === 'm2') elements.voteM2Btn.classList.add('selected');
                }
            }
        });
        
        if (changed) {
            applyFilters();
        }
    } catch (e) {
        console.error("Sync failed", e);
    }
}

// Set up polling
setInterval(syncVotes, 3000);

elements.voteM1Btn.addEventListener('click', () => saveVote('m1'));
elements.voteTieBtn.addEventListener('click', () => saveVote('tie'));
elements.voteM2Btn.addEventListener('click', () => saveVote('m2'));
elements.saveVoteBtn.addEventListener('click', () => saveVote());

elements.toggleGtBtn.addEventListener('click', () => {
    if (elements.gtPanel.style.display === 'none') {
        elements.gtPanel.style.display = 'flex';
    } else {
        elements.gtPanel.style.display = 'none';
    }
    setTimeout(() => {
        if (state.scale === 1 || state.scale > 0) {
            centerImages();
        }
    }, 10);
});

// Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
    // Prevent triggering shortcuts if typing in search or comment
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    if (!state.currentImg || state.filteredData.length === 0) return;
    
    const currentIndex = state.filteredData.findIndex(i => i.id === state.currentImg.id);
    
    if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (currentIndex > 0) selectImage(state.filteredData[currentIndex - 1]);
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (currentIndex < state.filteredData.length - 1) selectImage(state.filteredData[currentIndex + 1]);
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        saveVote('m1');
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        saveVote('m2');
    } else if (e.key === 'ArrowDown') { // Just mapping down arrow as well for tie?
        // Let's use T for tie maybe?
    } else if (e.key.toLowerCase() === 't') {
        e.preventDefault();
        saveVote('tie');
    } else if (e.key.toLowerCase() === 'm' && !state.showMask) {
        state.showMask = true;
        drawImages();
    } else if (e.key.toLowerCase() === 'n' && !state.showHeatmap) {
        state.showHeatmap = true;
        drawImages();
    } else if (e.key.toLowerCase() === 'b' && !state.showGT) {
        state.showGT = true;
        drawImages();
    }
});

window.addEventListener('keyup', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key.toLowerCase() === 'm') {
        state.showMask = false;
        drawImages();
    } else if (e.key.toLowerCase() === 'n') {
        state.showHeatmap = false;
        drawImages();
    } else if (e.key.toLowerCase() === 'b') {
        state.showGT = false;
        drawImages();
    }
});

// Init
fetchFolders();
