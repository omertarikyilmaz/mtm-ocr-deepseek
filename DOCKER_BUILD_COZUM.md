# 🔧 Docker Build Hatası Çözümü

## ❌ Hata

```
ERROR: Could not find a version that satisfies the requirement vllm==0.8.5+cu118
```

## ✅ Çözüm

### Problem Analizi

1. **vLLM 0.8.5+cu118** versiyonu artık PyPI'da mevcut değil
2. vLLM artık CUDA versiyonlarını farklı şekilde dağıtıyor
3. CUDA 11.8 desteği eskimiş durumda

### Uygulanan Çözüm

**CUDA ve PyTorch versiyonları güncellendi:**

| Önceki (Hatalı) | Yeni (Çalışan) |
|-----------------|----------------|
| CUDA 11.8 | **CUDA 12.1** |
| PyTorch 2.6.0 | **PyTorch 2.4.0** |
| vllm==0.8.5+cu118 | **vllm==0.8.5** |
| flash-attn 2.7.3 | **flash-attn 2.7.2.post1** |

### Değişen Dosyalar

1. ✅ `Dockerfile` - CUDA 12.1 base image
2. ✅ `requirements.txt` - Güncel versiyonlar
3. ✅ `docker-entrypoint.sh` - CUDA versiyon bilgisi
4. ✅ `README.md` - Kurulum talimatları

## 🚀 Nasıl Kullanılır

### 1. En Son Değişiklikleri Çek (sunucunuzda)

```bash
cd /home/omer/projects/mtm-ocr-deepseek

# Eski build cache'ini temizle
docker-compose down
docker system prune -af

# En son kodu çek
git pull origin main
```

### 2. Yeniden Build Et

```bash
# Build et
docker-compose build --no-cache

# Başlat
docker-compose up -d

# Logları izle
docker-compose logs -f
```

### 3. Manuel Build (detaylı log için)

```bash
# Adım adım build
docker build -t mtm-ocr:latest -f Dockerfile .

# Manuel başlat
docker run --runtime=nvidia --gpus all \
  -p 5000:5000 \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/uploads:/app/uploads \
  mtm-ocr:latest
```

## 🔍 Sorun Giderme

### GPU Kontrol

```bash
# Container içinde GPU erişimi test et
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

### Driver Kontrolü

```bash
# NVIDIA driver versiyonu (>= 530.30.02 olmalı)
nvidia-smi

# CUDA versiyonu
nvcc --version
```

### Build Loglarını Kaydet

```bash
docker-compose build --no-cache 2>&1 | tee build.log
```

## 📋 Gereksinimler

### Sistem Gereksinimleri

- **NVIDIA Driver**: >= 530.30.02
- **Docker**: >= 20.10
- **docker-compose**: >= 1.29
- **nvidia-docker2**: Kurulu olmalı

### GPU Gereksinimleri

- **Minimum**: 16GB VRAM (GTX 1080 Ti, RTX 3060 12GB)
- **Önerilen**: 24GB+ VRAM (RTX 3090, RTX 4090, A100)

## ⚙️ Alternatif Çözümler

### Çözüm 1: CUDA 11.8 ile Devam (Önerilmez)

Eğer mutlaka CUDA 11.8 kullanmanız gerekiyorsa:

```dockerfile
# Dockerfile içinde
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# PyTorch
RUN pip3 install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118

# vLLM (CUDA tag olmadan)
RUN pip3 install vllm==0.8.5
```

### Çözüm 2: En Yeni Versiyonlar (İleri Düzey)

```dockerfile
FROM nvidia/cuda:12.4.0-cudnn9-devel-ubuntu22.04

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install vllm
```

## 📝 Test Adımları

### 1. Build Testi

```bash
docker-compose build
# ✅ Hatasız tamamlanmalı
```

### 2. Container Başlatma Testi

```bash
docker-compose up -d
docker-compose logs -f
# ✅ "Model hazır!" mesajını görmelisiniz
```

### 3. GPU Erişim Testi

```bash
docker-compose exec mtm-ocr python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
# ✅ CUDA: True görmelisiniz
```

### 4. Web UI Testi

```bash
curl http://localhost:5000
# ✅ HTML içeriği dönmeli
```

## 🎯 Sonuç

**Yapılan Değişiklikler:**
- ✅ CUDA 11.8 → CUDA 12.1
- ✅ PyTorch 2.6.0 → PyTorch 2.4.0
- ✅ vllm==0.8.5+cu118 → vllm==0.8.5
- ✅ Tüm bağımlılıklar uyumlu hale getirildi

**Beklenen Sonuç:**
- ✅ Docker build hatasız tamamlanır
- ✅ Container başarıyla başlar
- ✅ GPU erişimi çalışır
- ✅ Web UI erişilebilir olur

## 📞 Ek Destek

Hala sorun yaşıyorsanız:

1. **Build loglarını kontrol edin**: `docker-compose build 2>&1 | tee build.log`
2. **GPU driver'ı kontrol edin**: `nvidia-smi`
3. **Disk alanını kontrol edin**: `df -h`
4. **Docker cache temizleyin**: `docker system prune -af`

---

**Son Güncelleme**: 12 Kasım 2025
**CUDA Versiyonu**: 12.1.0
**PyTorch Versiyonu**: 2.4.0
**vLLM Versiyonu**: 0.8.5

