#!/bin/bash
set -e

echo "========================================================================"
echo "MTM OCR - Medya Takip Merkezi"
echo "DeepSeek-OCR Docker Container"
echo "========================================================================"
echo ""

# CUDA kontrolü
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU tespit edildi:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[WARNING] NVIDIA GPU bulunamadi! CPU modunda calisacak"
fi

echo ""
echo "[INFO] Sistem ortami hazirlaniyor..."

# Python sürümü kontrolü
echo "       Python: $(python --version)"
echo "       PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "       CUDA: $(python -c 'import torch; print("Available (" + str(torch.version.cuda) + ")" if torch.cuda.is_available() else "Not Available")')"

# Gerekli dizinleri oluştur
mkdir -p /app/uploads
mkdir -p /app/output/results
mkdir -p /app/output/visualizations
mkdir -p /app/output/images

echo ""
echo "[INFO] Model yuklemesi (ilk calistirmada 5-10 dakika surebilir)"
echo "       Model: deepseek-ai/DeepSeek-OCR"
echo ""

# Model'i önceden indir (opsiyonel, hızlandırır)
python -c "
from transformers import AutoTokenizer
import os
os.environ['HF_HOME'] = '/root/.cache/huggingface'
try:
    print('[1/2] Tokenizer indiriliyor...')
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
    print('[2/2] Tokenizer yuklendi')
except Exception as e:
    print(f'[WARNING] Tokenizer yuklenemedi: {e}')
    print('          Model ilk kulanimda otomatik yuklenecek')
" || echo "[INFO] Model ilk kulanimda otomatik yuklenecek"

echo ""
echo "[INFO] Web uygulamasi baslatiliyor..."
echo "       Web arayuzu: http://localhost:5000"
echo "       API endpoint'leri:"
echo "         - POST /upload     : Dosya yukleme"
echo "         - POST /process    : OCR islemi"
echo "         - GET  /results    : Sonuclari listele"
echo "         - GET  /result/<id>: Tek sonuc detayi"
echo ""
echo "========================================================================"
echo ""

# Verilen komutu çalıştır
exec "$@"

