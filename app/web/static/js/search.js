/**
 * MTM OCR - Kelime Arama ve Vurgulama Modülü
 * JSON dosyalarında kelime arama ve görselde vurgulama
 */

let currentSearchResults = null;
let currentHighlightedCanvas = null;

/**
 * Kelime araması yap
 */
async function performKeywordSearch() {
    const keywordInput = document.getElementById('keywordInput');
    const searchResults = document.getElementById('searchResults');
    const searchStats = document.getElementById('searchStats');
    const searchResultsContainer = document.getElementById('searchResultsContainer');
    const searchBtn = document.getElementById('searchBtn');
    
    const keywords = keywordInput.value.trim();
    
    if (!keywords) {
        alert('Lütfen anahtar kelime girin!');
        return;
    }
    
    // Önceki sonuçları temizle
    currentSearchResults = null;
    currentHighlightedCanvas = null;
    searchResultsContainer.innerHTML = '';
    searchResults.style.display = 'none';
    
    searchBtn.disabled = true;
    searchBtn.textContent = '🔍 Aranıyor...';
    
    try {
        const response = await fetch('/api/search/keywords', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keywords: keywords, result_ids: [] })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            alert('Arama hatası: ' + (data.error || 'Bilinmeyen hata'));
            return;
        }
        
        displaySearchResults(data);
        
    } catch (error) {
        console.error('Search error:', error);
        alert('Arama hatası: ' + error.message);
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = '🔍 Ara ve Görselleştir';
    }
}

/**
 * Arama sonuçlarını göster
 */
function displaySearchResults(data) {
    const searchResults = document.getElementById('searchResults');
    const searchStats = document.getElementById('searchStats');
    const searchResultsContainer = document.getElementById('searchResultsContainer');
    
    searchResults.style.display = 'block';
    
    // İstatistikler
    searchStats.innerHTML = `
        <p><strong>🔍 Aranan Kelimeler:</strong> ${data.keywords.join(', ')}</p>
        <p><strong>📄 Taranan Dosya:</strong> ${data.total_files_searched}</p>
        <p><strong>✅ Eşleşme Bulunan:</strong> ${data.files_with_matches}</p>
    `;
    
    if (data.results.length === 0) {
        searchResultsContainer.innerHTML = '<p style="text-align: center; color: #999; padding: 40px;">Hiç eşleşme bulunamadı</p>';
        return;
    }
    
    // Sonuç kartları
    searchResultsContainer.innerHTML = data.results.map((result, idx) => {
        const keywordColors = generateKeywordColors(result.matches);
        
        return `
            <div class="search-result-card" onclick="showHighlightedResult(${idx})">
                <h4>📄 ${result.image_filename}</h4>
                <p class="result-info">✅ ${result.total_matches} eşleşme bulundu - <strong>Tıklayarak görselde görün</strong></p>
                <div class="keyword-legend">
                    ${result.matches.map(match => `
                        <span class="keyword-badge" style="background: ${keywordColors[match.keyword]};">
                            ${match.keyword} (${match.count})
                        </span>
                    `).join('')}
                </div>
                <div class="click-hint">🖼️ Görseli vurgulu olarak görmek için tıklayın</div>
            </div>
        `;
    }).join('');
    
    currentSearchResults = data.results;
}

/**
 * Her keyword için renk üret
 */
function generateKeywordColors(matches) {
    const colors = {};
    matches.forEach((match, i) => {
        const hue = (i * 137.5) % 360;
        colors[match.keyword] = `hsl(${hue}, 70%, 50%)`;
    });
    return colors;
}

/**
 * Vurgulu görseli modal'da göster
 */
function showHighlightedResult(resultIndex) {
    const result = currentSearchResults[resultIndex];
    if (!result) {
        alert('Sonuç bulunamadı');
        return;
    }
    
    const modal = document.getElementById('resultModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');
    const keywordColors = generateKeywordColors(result.matches);
    
    modalTitle.textContent = `🔍 ${result.image_filename}`;
    
    modalBody.innerHTML = `
        <div class="modal-section">
            <h3>Bulunan Kelimeler</h3>
            <div class="keyword-legend">
                ${result.matches.map(match => `
                    <span class="keyword-badge" style="background: ${keywordColors[match.keyword]};">
                        ${match.keyword} (${match.count} eşleşme)
                    </span>
                `).join('')}
            </div>
            <p class="info-text">Toplam ${result.total_matches} eşleşme görselde kutu içinde vurgulanmıştır</p>
        </div>
        
        <div class="modal-section" style="text-align: center;">
            <button class="btn btn-success" onclick="downloadHighlightedImage('${result.image_filename}')">
                📥 Vurgulu Görseli İndir
            </button>
            <p class="info-text">Kelimeleri kutu içinde vurgulu haliyle indirilir</p>
        </div>
        
        <div id="canvas-loading" class="loading-container">
            <div class="spinner"></div>
            <p>Görsel yükleniyor ve kelimeler vurgulanıyor...</p>
        </div>
        
        <div id="canvas-container" class="canvas-container" style="display: none;">
            <canvas id="modal-canvas" class="highlighted-canvas"></canvas>
        </div>
        
        <div class="modal-section stats">
            <p><strong>Görsel Boyutu:</strong> ${result.image_size.width} x ${result.image_size.height} px</p>
            <p><strong>Tarih:</strong> ${result.timestamp}</p>
            <p><strong>Toplam Eşleşme:</strong> ${result.total_matches}</p>
        </div>
    `;
    
    modal.classList.add('active');
    
    setTimeout(() => drawHighlightedImage(result, keywordColors), 100);
}

/**
 * Vurgulu görseli canvas'a çiz - BASİT VE DOĞRU
 */
function drawHighlightedImage(result, keywordColors) {
    const canvas = document.getElementById('modal-canvas');
    const canvasContainer = document.getElementById('canvas-container');
    const loadingDiv = document.getElementById('canvas-loading');
    
    if (!canvas) {
        console.error('Canvas bulunamadı');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onerror = () => {
        loadingDiv.innerHTML = '<p style="color: red;">❌ Görsel yüklenemedi</p>';
    };
    
    img.onload = () => {
        console.log('=== GÖRSEL YÜKLEME ===');
        console.log('Orijinal görsel:', img.width, 'x', img.height);
        console.log('JSON image_size:', result.image_size);
        
        // Canvas boyutunu ayarla
        const maxWidth = 1100;
        const scale = Math.min(1, maxWidth / img.width);
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        
        console.log('Canvas boyutu:', canvas.width, 'x', canvas.height);
        console.log('Scale faktörü:', scale);
        
        // Görseli çiz
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // Kutuları çiz
        let totalBoxes = 0;
        result.matches.forEach((match) => {
            const color = keywordColors[match.keyword];
            console.log(`\n=== KELİME: "${match.keyword}" (${match.positions.length} eşleşme) ===`);
            
            match.positions.forEach((pos, idx) => {
                const bbox = pos.bbox;
                
                if (!bbox) {
                    console.warn(`Bbox yok, atlandı`);
                    return;
                }
                
                // Backend bbox'u ZATEN orijinal piksel cinsinden
                // Canvas scale edilmiş, bbox'ları da scale et
                const x = bbox.x1 * scale;
                const y = bbox.y1 * scale;
                const w = (bbox.x2 - bbox.x1) * scale;
                const h = (bbox.y2 - bbox.y1) * scale;
                
                if (idx < 2) {
                    console.log(`[${idx}] "${pos.text}"`);
                    console.log(`  Orijinal bbox: x1=${bbox.x1}, y1=${bbox.y1}, x2=${bbox.x2}, y2=${bbox.y2}`);
                    console.log(`  Çizilen: x=${Math.round(x)}, y=${Math.round(y)}, w=${Math.round(w)}, h=${Math.round(h)}`);
                }
                
                if (w <= 0 || h <= 0) {
                    console.warn(`Geçersiz boyut, atlandı`);
                    return;
                }
                
                // Kutuyu çiz
                ctx.strokeStyle = color;
                ctx.lineWidth = 4;
                ctx.strokeRect(x, y, w, h);
                
                ctx.fillStyle = color.replace(')', ', 0.2)').replace('hsl', 'hsla');
                ctx.fillRect(x, y, w, h);
                
                totalBoxes++;
            });
        });
        
        console.log(`\n=== TOPLAM ÇİZİLEN KUTU: ${totalBoxes} ===`);
        
        loadingDiv.style.display = 'none';
        canvasContainer.style.display = 'block';
        currentHighlightedCanvas = canvas;
    };
    
    // Base64 görseli yükle
    const base64 = result.image_base64;
    if (base64.startsWith('data:image')) {
        img.src = base64;
    } else {
        img.src = 'data:image/jpeg;base64,' + base64;
    }
}

/**
 * Vurgulu görseli indir
 */
function downloadHighlightedImage(filename) {
    if (!currentHighlightedCanvas) {
        alert('Görsel henüz yüklenmedi');
        return;
    }
    
    currentHighlightedCanvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename.replace(/\.(jpg|png)$/i, '_vurgulu.jpg');
        a.click();
        URL.revokeObjectURL(url);
    }, 'image/jpeg', 0.95);
}

