/**
 * MTM OCR - Ortak Fonksiyonlar
 */

/**
 * Tab değiştirme
 */
function switchTab(tabName) {
    // Tüm tabları gizle
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Tüm butonlardan active sınıfını kaldır
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Seçili tabı göster
    if (tabName === 'ocr') {
        document.getElementById('ocrTab').classList.add('active');
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
    } else if (tabName === 'search') {
        document.getElementById('searchTab').classList.add('active');
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
    }
}

/**
 * Modal kapat
 */
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('resultModal');
    const modalClose = document.getElementById('modalClose');
    
    if (modalClose) {
        modalClose.addEventListener('click', () => {
            modal.classList.remove('active');
        });
    }
    
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    }
});

