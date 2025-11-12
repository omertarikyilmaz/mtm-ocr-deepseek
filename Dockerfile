# MTM OCR - Medya Takip Merkezi
# Docker image for DeepSeek-OCR with Web UI

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Temel paketler
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python sembolik bağlantı
RUN ln -s /usr/bin/python3 /usr/bin/python

# Çalışma dizini
WORKDIR /app

# CUDA ortam değişkenleri
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TRITON_PTXAS_PATH=/usr/local/cuda-12.1/bin/ptxas

# Python bağımlılıkları - önce temel paketler
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch ve torchvision (CUDA 12.1 için)
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# vLLM kurulumu (CUDA 12.1 uyumlu)
RUN pip3 install --no-cache-dir vllm==0.8.5

# DeepSeek OCR kodu - önce dizinleri oluştur
RUN mkdir -p /app/deepseek_vllm/process /app/deepseek_vllm/deepencoder

# DeepSeek-OCR-vllm dosyalarını kopyala
COPY DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/*.py /app/deepseek_vllm/
COPY DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ /app/deepseek_vllm/process/
COPY DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/ /app/deepseek_vllm/deepencoder/

# __init__.py oluştur
RUN echo 'from .deepseek_ocr import DeepseekOCRForCausalLM' > /app/deepseek_vllm/__init__.py && \
    echo '__all__ = ["DeepseekOCRForCausalLM"]' >> /app/deepseek_vllm/__init__.py

# PYTHONPATH ayarla (hem /app hem de /app/deepseek_vllm)
ENV PYTHONPATH="/app:/app/deepseek_vllm:${PYTHONPATH}"

# Diğer bağımlılıklar
RUN pip3 install --no-cache-dir \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    PyMuPDF \
    img2pdf \
    einops \
    easydict \
    addict \
    Pillow \
    numpy \
    flask \
    tqdm

# Flash Attention (opsiyonel, hızlandırır) - CUDA 12.1 için
RUN pip3 install --no-cache-dir flash-attn==2.7.2.post1 --no-build-isolation || echo "Flash attention kurulumu başarısız, devam ediliyor..."

# Uygulama dosyaları
COPY mtm_batch_ocr.py /app/
COPY web_ui.py /app/
COPY templates/ /app/templates/

# Gerekli dizinler
RUN mkdir -p /app/uploads /app/output /app/models

# Port
EXPOSE 5000

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Başlangıç scripti
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "web_ui.py", "--host", "0.0.0.0", "--port", "5000"]

