FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python ve sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11'i default yap
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# pip güncelle
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch (CUDA 12.1)
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# vLLM
RUN pip install vllm==0.8.5

# Diğer bağımlılıklar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# DeepSeek OCR kodları
COPY . .

# Port
EXPOSE 8000

# Başlat
CMD ["python", "app.py"]

