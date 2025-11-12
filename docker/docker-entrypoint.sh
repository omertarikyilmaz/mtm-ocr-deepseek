#!/bin/bash
set -e

echo "========================================================================"
echo "MTM OCR - Medya Takip Merkezi"
echo "DeepSeek-OCR Docker Container"
echo "========================================================================"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU tespit edildi"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[UYARI] NVIDIA GPU bulunamadi, CPU modunda calisacak"
fi

echo ""
echo "[INFO] Sistem ortami hazirlaniyor"
echo "       Python: $(python --version)"
echo "       PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "       CUDA: $(python -c 'import torch; print("Mevcut" if torch.cuda.is_available() else "Mevcut Degil")')"

mkdir -p /app/uploads
mkdir -p /app/output/results
mkdir -p /app/output/originals

echo ""
echo "[INFO] Model yuklemesi baslatiliyor"
echo "       Model: deepseek-ai/DeepSeek-OCR"

python -c "
from transformers import AutoTokenizer
import os
os.environ['HF_HOME'] = '/root/.cache/huggingface'
try:
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
    print('[INFO] Tokenizer hazir')
except Exception as e:
    print('[UYARI] Tokenizer yuklenemedi, model ilk kullanımda yuklenecek')
" || echo "[INFO] Model ilk kullanımda otomatik yuklenecek"

echo ""
echo "[INFO] Web uygulamasi baslatiliyor"
echo "       Adres: http://localhost:5000"
echo "========================================================================"
echo ""

exec "$@"

