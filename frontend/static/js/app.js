/**
 * MTM OCR Frontend - Basit OCR İşleme
 * Bbox mantığı kaldırılmış, sadece metin işleme
 */

const API_BASE = '/api';
let uploadedFiles = [];
let processingInterval = null;

// Sayfa yüklendiğinde
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    loadResults();
});

/**
 * Uygulamayı başlat
 */
function initApp() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const clearBtn = document.getElementById('clearBtn');
    const processBtn = document.getElementById('processBtn');
    const refreshBtn = document.getElementById('refreshBtn');
    const deleteAllBtn = document.getElementById('deleteAllBtn');
    const modalClose = document.getElementById('modalClose');
    
    // Upload zone events
    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
    
    // Button events
    clearBtn.addEventListener('click', clearFiles);
    processBtn.addEventListener('click', processImages);
    refreshBtn.addEventListener('click', loadResults);
    deleteAllBtn.addEventListener('click', deleteAllResults);
    modalClose.addEventListener('click', closeModal);
}

/**
 * Dosya yükleme
 */
async function handleFiles(files) {
    const formData = new FormData();
    
    for (let file of files) {
        formData.append('files[]', file);
    }
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            uploadedFiles = uploadedFiles.concat(data.files);
            renderFileList();
            document.getElementById('processBtn').disabled = false;
            document.getElementById('clearBtn').disabled = false;
        } else {
            alert('Hata: ' + data.error);
        }
    } catch (error) {
        alert('Dosya yükleme hatası: ' + error);
    }
}

/**
 * Dosya listesini render et
 */
function renderFileList() {
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = uploadedFiles.map((file, index) => `
        <div class="file-item">
            <span class="file-name">${file.original_name}</span>
            <button class="file-remove" onclick="removeFile(${index})">Kaldır</button>
        </div>
    `).join('');
}

/**
 * Dosya kaldır
 */
function removeFile(index) {
    uploadedFiles.splice(index, 1);
    renderFileList();
    
    if (uploadedFiles.length === 0) {
        document.getElementById('processBtn').disabled = true;
        document.getElementById('clearBtn').disabled = true;
    }
}

/**
 * Tüm dosyaları temizle
 */
function clearFiles() {
    uploadedFiles = [];
    renderFileList();
    document.getElementById('processBtn').disabled = true;
    document.getElementById('clearBtn').disabled = true;
    document.getElementById('statusBox').classList.remove('active');
}

/**
 * OCR işlemini başlat
 */
async function processImages() {
    if (uploadedFiles.length === 0) return;
    
    const filenames = uploadedFiles.map(f => f.filename);
    const processBtn = document.getElementById('processBtn');
    const statusBox = document.getElementById('statusBox');
    const statusMessage = document.getElementById('statusMessage');
    const progressFill = document.getElementById('progressFill');
    
    try {
        processBtn.disabled = true;
        statusBox.classList.add('active');
        statusMessage.textContent = 'OCR işlemi başlatılıyor...';
        progressFill.style.width = '0%';
        progressFill.textContent = '0%';
        
        const response = await fetch(`${API_BASE}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filenames })
        });
        
        const data = await response.json();
        
        if (data.success) {
            startStatusPolling();
        } else {
            alert('Hata: ' + data.error);
            processBtn.disabled = false;
        }
    } catch (error) {
        alert('İşlem başlatma hatası: ' + error);
        processBtn.disabled = false;
    }
}

/**
 * İşlem durumunu kontrol et
 */
function startStatusPolling() {
    const statusMessage = document.getElementById('statusMessage');
    const progressFill = document.getElementById('progressFill');
    const processBtn = document.getElementById('processBtn');
    
    processingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/status`);
            const status = await response.json();
            
            statusMessage.textContent = status.status_message || 'İşleniyor...';
            
            if (status.total > 0) {
                const progress = Math.round((status.progress / status.total) * 100);
                progressFill.style.width = progress + '%';
                progressFill.textContent = progress + '%';
            }
            
            if (!status.is_processing) {
                clearInterval(processingInterval);
                processBtn.disabled = false;
                setTimeout(loadResults, 1000);
            }
        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 1000);
}

/**
 * Sonuçları yükle
 */
async function loadResults() {
    const resultsGrid = document.getElementById('resultsGrid');
    
    try {
        const response = await fetch(`${API_BASE}/results`);
        const data = await response.json();
        
        if (data.results.length === 0) {
            resultsGrid.innerHTML = '<p style="color: #999; text-align: center; padding: 40px;">Henüz işlenmiş gazete yok</p>';
            return;
        }
        
        resultsGrid.innerHTML = data.results.map(result => `
            <div class="result-card" onclick="showResult('${result.id}')">
                <div class="result-card-delete" onclick="event.stopPropagation(); deleteResult('${result.id}')">×</div>
                <div class="result-image">📄</div>
                <div class="result-info">
                    <div class="result-title">${result.filename}</div>
                    <div class="result-meta">
                        <span>${result.text_length} karakter</span>
                        <span>${result.timestamp}</span>
                    </div>
                    <div class="result-actions">
                        <button class="btn btn-small" onclick="event.stopPropagation(); downloadFile('${result.id}')">
                            JSON İndir
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Results loading error:', error);
    }
}

/**
 * Sonuç detaylarını göster
 */
async function showResult(resultId) {
    const modal = document.getElementById('resultModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');
    
    try {
        const response = await fetch(`${API_BASE}/result/${resultId}`);
        const data = await response.json();
        
        modalTitle.textContent = data.image_filename;
        
        // Görseli base64'ten göster
        let imageHtml = '';
        if (data.image_base64) {
            const imgSrc = data.image_base64.startsWith('data:') 
                ? data.image_base64 
                : `data:image/jpeg;base64,${data.image_base64}`;
            imageHtml = `
                <div class="modal-section">
                    <h3>Görsel</h3>
                    <img src="${imgSrc}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />
                </div>
            `;
        }
        
        modalBody.innerHTML = `
            <h3 style="margin-bottom: 15px;">İstatistikler</h3>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <p><strong>Metin Uzunluğu:</strong> ${data.text.length} karakter</p>
                <p><strong>Görsel Boyutu:</strong> ${data.image_size.width} x ${data.image_size.height} px</p>
                <p><strong>Tarih:</strong> ${data.timestamp}</p>
            </div>
            
            ${imageHtml}
            
            <h3 style="margin-bottom: 15px;">OCR Metni</h3>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; margin-bottom: 20px; font-family: monospace;">
                ${data.text}
            </div>
        `;
        
        modal.classList.add('active');
    } catch (error) {
        alert('Sonuç yükleme hatası: ' + error);
    }
}

/**
 * Modal'ı kapat
 */
function closeModal() {
    document.getElementById('resultModal').classList.remove('active');
}

/**
 * JSON dosyasını indir
 */
function downloadFile(resultId) {
    window.location.href = `${API_BASE}/download/${resultId}`;
}

/**
 * Sonucu sil
 */
async function deleteResult(resultId) {
    if (!confirm('Bu sonucu silmek istediğinizden emin misiniz?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/delete/${resultId}`, { method: 'DELETE' });
        const data = await response.json();
        
        if (data.success) {
            loadResults();
        } else {
            alert('Silme hatası: ' + (data.error || 'Bilinmeyen hata'));
        }
    } catch (error) {
        alert('Silme hatası: ' + error);
    }
}

/**
 * Tüm sonuçları sil
 */
async function deleteAllResults() {
    if (!confirm('TÜM sonuçları silmek istediğinizden emin misiniz?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/delete-all`, { method: 'DELETE' });
        const data = await response.json();
        
        if (data.success) {
            loadResults();
        } else {
            alert('Silme hatası: ' + (data.error || 'Bilinmeyen hata'));
        }
    } catch (error) {
        alert('Silme hatası: ' + error);
    }
}

