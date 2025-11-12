# 🔧 NVIDIA Runtime Hatası Çözümü

## ❌ Hata Mesajı

```
error response from daemon: unknown or invalid runtime name: nvidia
```

## 📋 Problem

Docker, NVIDIA GPU'yu kullanmak için gerekli `nvidia` runtime'ını bulamıyor.

## ✅ HIZLI ÇÖZÜM (ÖNERİLEN)

### Çözüm 1: Güncellenmiş docker-compose.yml Kullan

Docker-compose.yml dosyası güncellendi. Modern GPU sözdizimini kullanıyor:

```bash
cd /home/omer/projects/mtm-ocr-deepseek

# En son değişiklikleri çek
git pull origin main

# Direkt çalıştır
docker-compose up -d
```

**Artık `runtime: nvidia` yerine modern `deploy.resources` sözdizimi kullanılıyor!**

---

## 🔧 DETAYLI ÇÖZÜMLER

### Çözüm 2: nvidia-docker2 Kurulumu (Eğer yoksa)

```bash
# 1. NVIDIA Docker repository ekle
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Kurulum
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-docker2

# 3. Docker'ı yeniden başlat
sudo systemctl restart docker

# 4. Test et
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Çözüm 3: Docker Daemon Yapılandırması

Eğer nvidia-docker2 kuruluysa ama çalışmıyorsa:

```bash
# 1. Docker daemon.json dosyasını düzenle
sudo nano /etc/docker/daemon.json

# 2. Aşağıdaki içeriği ekle/güncelle:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}

# 3. Kaydet ve Docker'ı restart et
sudo systemctl restart docker

# 4. Kontrol et
docker info | grep -i runtime
```

### Çözüm 4: Manuel GPU Mapping (runtime olmadan)

```bash
# --runtime yerine --gpus kullan
docker run --rm --gpus all \
  -p 5000:5000 \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/uploads:/app/uploads \
  mtm-ocr:latest
```

---

## 🚀 SUNUCUNUZDA YAPMANIZ GEREKENLER

### Seçenek A: En Kolay (Güncel docker-compose.yml ile)

```bash
cd /home/omer/projects/mtm-ocr-deepseek

# 1. En son kodu çek
git pull origin main

# 2. Çalıştır (artık runtime: nvidia kullanmıyor)
docker-compose up -d

# 3. Logları kontrol et
docker-compose logs -f
```

### Seçenek B: nvidia-docker2 Kur (Kalıcı çözüm)

```bash
# 1. nvidia-container-toolkit kur
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-docker2

# 2. Docker restart
sudo systemctl restart docker

# 3. Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 4. Projeni başlat
cd /home/omer/projects/mtm-ocr-deepseek
docker-compose up -d
```

---

## 🔍 Kontrol Komutları

### GPU Erişimi Test

```bash
# 1. nvidia-smi (host)
nvidia-smi

# 2. Docker ile GPU test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 3. Container içinde test
docker-compose exec mtm-ocr nvidia-smi
```

### Docker Runtime Kontrolü

```bash
# Docker info
docker info | grep -i runtime

# nvidia-docker versiyonu
nvidia-docker version

# nvidia-container-toolkit versiyonu
nvidia-container-toolkit --version
```

### Log Kontrolleri

```bash
# Docker logs
sudo journalctl -u docker -n 50

# Container logs
docker-compose logs -f mtm-ocr
```

---

## ⚠️ Sık Karşılaşılan Sorunlar

### Sorun 1: "nvidia-smi not found in container"
```bash
# Çözüm: Base image CUDA içermeli
# Dockerfile'da: FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 ✅
```

### Sorun 2: "no NVIDIA GPU devices found"
```bash
# Kontrol et
lspci | grep -i nvidia
nvidia-smi

# Driver kur
sudo apt install nvidia-driver-535
sudo reboot
```

### Sorun 3: "permission denied"
```bash
# Docker grubuna ekle
sudo usermod -aG docker $USER
newgrp docker

# veya
sudo chmod 666 /var/run/docker.sock
```

### Sorun 4: "Docker daemon not responding"
```bash
sudo systemctl status docker
sudo systemctl restart docker
```

---

## 📊 Versiyon Uyumluluğu

| Component | Minimum Versiyon | Önerilen |
|-----------|------------------|----------|
| Docker | 19.03+ | 24.0+ |
| Docker Compose | 1.28+ | 2.20+ |
| NVIDIA Driver | 530.30.02+ | 535+ |
| nvidia-docker2 | 2.13+ | Latest |
| CUDA | 12.0+ | 12.1 |

---

## ✅ Başarı Kontrolü

Container başarıyla başladıysa:

```bash
# 1. Container çalışıyor
docker ps | grep mtm-ocr

# 2. GPU erişimi var
docker-compose exec mtm-ocr nvidia-smi

# 3. CUDA çalışıyor
docker-compose exec mtm-ocr python -c "import torch; print(torch.cuda.is_available())"

# 4. Web UI erişilebilir
curl http://localhost:5000

# Tarayıcıda: http://sunucu-ip:5000
```

---

## 🎯 Özet

**En Hızlı Çözüm:**
```bash
cd /home/omer/projects/mtm-ocr-deepseek
git pull origin main
docker-compose up -d
```

**Kalıcı Çözüm (nvidia-docker2 kur):**
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 📞 Hala Sorun mu Var?

1. **Driver kontrol**: `nvidia-smi`
2. **Docker servis**: `sudo systemctl status docker`
3. **GPU test**: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
4. **Loglar**: `docker-compose logs -f`

**Son Güncelleme**: 12 Kasım 2025

