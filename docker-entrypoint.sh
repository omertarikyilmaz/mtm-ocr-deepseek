#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          MTM OCR - Medya Takip Merkezi                  ║"
echo "║          DeepSeek-OCR Docker Container                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# CUDA kontrolü
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU bulundu:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  NVIDIA GPU bulunamadı! CPU modunda çalışacak (çok yavaş olabilir)"
fi

echo ""
echo "🔧 Ortam hazırlanıyor..."

# Python sürümü kontrolü
echo "Python sürümü: $(python --version)"
echo "PyTorch sürümü: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA erişimi: $(python -c 'import torch; print("Evet (" + str(torch.version.cuda) + ")" if torch.cuda.is_available() else "Hayır")')"

# Gerekli dizinleri oluştur
mkdir -p /app/uploads
mkdir -p /app/output/results
mkdir -p /app/output/visualizations
mkdir -p /app/output/images

echo ""
echo "📦 Model indiriliyor (ilk çalıştırmada birkaç dakika sürebilir)..."
echo "   Model: deepseek-ai/DeepSeek-OCR"
echo ""

# Model'i önceden indir (opsiyonel, hızlandırır)
python -c "
from transformers import AutoTokenizer, AutoModel
import os
os.environ['HF_HOME'] = '/root/.cache/huggingface'
try:
    print('📥 Tokenizer indiriliyor...')
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
    print('✅ Tokenizer hazır')
except Exception as e:
    print(f'⚠️  Tokenizer indirilemedi: {e}')
    print('   Model ilk kullanımda otomatik indirilecek')
" || echo "Model ilk kullanımda otomatik indirilecek"

echo ""
echo "🚀 Uygulama başlatılıyor..."
echo "   Web arayüzü: http://localhost:5000"
echo "   API endpoint'leri:"
echo "     - POST /upload - Dosya yükleme"
echo "     - POST /process - OCR işlemi"
echo "     - GET /results - Sonuçları listele"
echo "     - GET /result/<id> - Tek sonuç detayı"
echo ""
echo "📝 Loglara devam ediliyor..."
echo "══════════════════════════════════════════════════════════"
echo ""

# Verilen komutu çalıştır
exec "$@"

