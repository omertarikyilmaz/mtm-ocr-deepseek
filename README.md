# MTM OCR - Medya Takip Merkezi

DeepSeek OCR tabanlı, basit ve temiz OCR servisi. GPU destekli, Docker ile kolay kurulum.

## 🏗️ Mimari

```
mtm-ocr-deepseek/
├── backend/          # Flask API (Python)
├── frontend/         # Web UI (HTML/CSS/JS + Nginx)
├── deepseek/         # DeepSeek OCR Servisi (GPU)
├── docker-compose.yml
├── start.sh          # Başlat
├── stop.sh           # Durdur
└── update.sh         # GitHub'tan güncelle + build
```

### Servisler

1. **Frontend** (Port 80): Nginx ile statik web arayüzü
2. **Backend** (Port 5000): Flask API - Dosya yönetimi
3. **DeepSeek** (Port 8000): OCR işleme servisi (GPU gerekli)

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Docker (v20.10+) with Compose plugin
- NVIDIA GPU (CUDA 12.1 uyumlu)
- NVIDIA Docker Runtime

### Kurulum

```bash
# 1. Projeyi klonla
git clone https://github.com/omertarikyilmaz/mtm-ocr-deepseek.git
cd mtm-ocr-deepseek

# 2. Başlat (ilk seferde model indirilir, ~10-15 dakika sürebilir)
./start.sh
```

### Kullanım

```bash
# Başlat
./start.sh

# Durdur
./stop.sh

# GitHub'tan güncelle (sadece build, başlatmaz)
./update.sh

# Logları görüntüle
docker compose logs -f

# Tek bir servisin logları
docker compose logs -f deepseek
```

## 🌐 Erişim

- **Web Arayüzü**: http://localhost
- **Backend API**: http://localhost:5000
- **DeepSeek OCR**: http://localhost:8000

## 📂 Klasör Yapısı

```
├── uploads/          # Yüklenen görseller
├── output/
│   └── results/      # OCR sonuçları (JSON)
└── models/           # HuggingFace model önbelleği
```

## 🔧 Yapılandırma

### GPU Ayarları

`docker-compose.yml` içinde GPU ayarlarını değiştirebilirsiniz:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0  # GPU ID
  - CUDA_VISIBLE_DEVICES=0
```

### Bellek Ayarları

```yaml
shm_size: '8gb'  # Paylaşılan bellek (artırılabilir)
```

## 📝 API Dökümantasyonu

### Backend API

```bash
# Dosya yükle
POST /api/upload
Content-Type: multipart/form-data
Body: files[]

# OCR işle
POST /api/process
Content-Type: application/json
Body: {"filenames": ["uuid.jpg", ...]}

# Sonuçları listele
GET /api/results

# Tek sonuç
GET /api/result/{id}

# JSON indir
GET /api/download/{id}

# Sil
DELETE /api/delete/{id}
DELETE /api/delete-all
```

## 🛠️ Geliştirme

### Tek bir servisi yeniden build et

```bash
docker compose build backend
docker compose build frontend
docker compose build deepseek
```

### Container'a gir

```bash
docker compose exec backend bash
docker compose exec deepseek bash
```

### Model önbelleğini temizle

```bash
rm -rf models/*
```

## 📊 Performans

- **İlk başlatma**: ~10-15 dakika (model indirme)
- **Sonraki başlatmalar**: ~2-3 dakika (model yükleme)
- **OCR hızı**: ~10-30 saniye/sayfa (GPU'ya göre değişir)

## 🔍 Sorun Giderme

### Model yüklenmiyor

```bash
# DeepSeek servisini yeniden başlat
docker compose restart deepseek

# Logları kontrol et
docker compose logs -f deepseek
```

### GPU erişim hatası

```bash
# NVIDIA runtime kontrolü
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Docker Compose GPU desteği
docker compose config | grep -A 5 "devices"
```

### Port çakışması

`docker-compose.yml` içinde portları değiştirin:

```yaml
ports:
  - "8080:80"      # Frontend (80 yerine 8080)
  - "5001:5000"    # Backend
```

## 📄 Lisans

MIT License - Detaylar için `LICENSE` dosyasına bakın.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing`)
3. Commit edin (`git commit -m 'feat: add amazing feature'`)
4. Push edin (`git push origin feature/amazing`)
5. Pull Request açın

## 📧 İletişim

Sorularınız için: https://github.com/omertarikyilmaz/mtm-ocr-deepseek/issues
