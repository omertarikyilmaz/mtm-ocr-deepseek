# 📰 MTM OCR - Medya Takip Merkezi

DeepSeek-OCR tabanlı, birden fazla gazete sayfasını aynı anda işleyip **her kelimenin pozisyonunu** kayıt eden batch OCR sistemi.

## ✨ Özellikler

- **🚀 Batch İşleme**: Birden fazla gazete sayfasını aynı anda OCR ile okuma
- **📍 Kelime Pozisyonları**: Her kelimenin piksel koordinatlarını JSON formatında kaydetme
- **🎨 Görselleştirme**: Bounding box'larla kelime pozisyonlarını görsel üzerine çizme
- **🌐 Web Arayüzü**: Drag & drop ile dosya yükleme ve sonuçları görüntüleme
- **🐳 Docker Desteği**: Tek komutla çalışır hale getirme
- **⚡ GPU Hızlandırma**: NVIDIA GPU desteği ile hızlı işleme
- **📊 Detaylı Raporlar**: JSON, TXT ve görselleştirilmiş çıktılar

## 🎯 Kullanım Senaryoları

- Gazete arşivlerinin dijitalleştirilmesi
- Medya takip ve analiz sistemleri
- Gazete sayfalarından metin ve konum çıkarma
- OCR sonuçlarının pozisyon bilgisiyle birlikte saklanması

## 🚀 Hızlı Başlangıç (Docker ile - ÖNERİLEN)

### Ön Gereksinimler

1. **Docker ve Docker Compose** yüklü olmalı
2. **NVIDIA GPU** ve **nvidia-docker** runtime yüklü olmalı
   ```bash
   # NVIDIA Docker runtime kurulumu (Ubuntu/Debian)
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### Tek Komutla Başlatma

```bash
# 1. Projeyi klonlayın
cd /home/ower/Projects/mtm-ocr-deepseek

# 2. Docker container'ı başlatın (ilk çalıştırmada model indirilecek, 10-15 dk sürebilir)
docker-compose up -d

# 3. Logları takip edin
docker-compose logs -f

# 4. Web arayüzünü açın
# Tarayıcıda: http://localhost:5000
```

İlk çalıştırmada:
- Docker image'ı build edilecek (~5 dakika)
- DeepSeek-OCR modeli indirilecek (~15GB, ilk kulanımda)
- Uygulama otomatik başlayacak

### Durdurma ve Yönetim

```bash
# Container'ı durdurma
docker-compose down

# Yeniden başlatma
docker-compose restart

# Logları görüntüleme
docker-compose logs -f mtm-ocr

# Container içine giriş
docker-compose exec mtm-ocr bash

# Yeniden build (kod değişikliklerinden sonra)
docker-compose build --no-cache
docker-compose up -d
```

## 💻 Manuel Kurulum (Docker olmadan)

### Ön Gereksinimler

- Python 3.10+
- CUDA 12.1 (veya CUDA 11.8+)
- NVIDIA GPU (en az 24GB VRAM önerilir)
- NVIDIA Driver >= 530.30.02

### Kurulum Adımları

```bash
# 1. Python ortamı oluştur
conda create -n mtm-ocr python=3.10 -y
conda activate mtm-ocr

# 2. PyTorch kur (CUDA 12.1)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 3. vLLM kur
pip install vllm==0.8.5

# 4. Diğer bağımlılıklar
pip install -r requirements.txt

# 5. Flash Attention (opsiyonel, performans için)
pip install flash-attn==2.7.2.post1 --no-build-isolation

# 6. Web UI'ı başlat
python web_ui.py --host 0.0.0.0 --port 5000
```

## 📖 Kullanım Kılavuzu

### 1. Web Arayüzü ile Kullanım

1. **Tarayıcıda aç**: http://localhost:5000
2. **Dosya yükle**: Gazete sayfalarını drag & drop veya dosya seç
3. **İşleme başlat**: "OCR İşlemini Başlat" butonuna tıkla
4. **Sonuçları görüntüle**: İşlem bitince aşağıda sonuçlar görünür
5. **Detaylara bak**: Sonuç kartlarına tıklayarak detayları görüntüle
6. **İndir**: JSON, TXT veya görsel olarak indir

### 2. Komut Satırı ile Kullanım

```bash
# Tek klasördeki tüm görselleri işle
python mtm_batch_ocr.py --input ./gazeteler/ --output ./sonuclar/

# Belirli bir pattern ile
python mtm_batch_ocr.py --input "./gazeteler/*.jpg" --output ./sonuclar/

# Gelişmiş parametreler
python mtm_batch_ocr.py \
  --input ./gazeteler/ \
  --output ./sonuclar/ \
  --model deepseek-ai/DeepSeek-OCR \
  --device 0 \
  --concurrency 50 \
  --workers 32
```

### 3. Python API ile Kullanım

```python
from mtm_batch_ocr import MTMOCRProcessor

# Processor'ı başlat
processor = MTMOCRProcessor(
    output_dir="./output",
    max_concurrency=50
)

# Görselleri işle
results = processor.process_batch(
    image_paths=["gazete1.jpg", "gazete2.jpg", "gazete3.jpg"],
    num_workers=32
)

# Sonuçları kullan
for result in results:
    print(f"Dosya: {result['image_filename']}")
    print(f"Kelime sayısı: {result['word_count']}")
    
    # Her kelimenin pozisyonu
    for word in result['words']:
        print(f"  '{word['text']}' -> x:{word['bbox']['x1']}, y:{word['bbox']['y1']}")
```

## 📂 Çıktı Formatı

Her işlenen gazete için 3 dosya oluşturulur:

### 1. JSON Dosyası (tam bilgi)
```json
{
  "image_path": "gazete1.jpg",
  "image_filename": "gazete1",
  "timestamp": "20250112_143022",
  "image_size": {
    "width": 2480,
    "height": 3508
  },
  "word_count": 1247,
  "words": [
    {
      "text": "Başlık",
      "bbox": {
        "x1": 245,
        "y1": 120,
        "x2": 380,
        "y2": 155,
        "width": 135,
        "height": 35
      },
      "normalized_bbox": {
        "x1": 98,
        "y1": 34,
        "x2": 153,
        "y2": 44
      },
      "index": 0
    }
  ],
  "full_text": "Temiz metin içeriği...",
  "raw_ocr_output": "Ham OCR çıktısı (taglar ile)..."
}
```

### 2. TXT Dosyası (sadece metin)
```
Temiz, okunabilir metin formatı
Tüm taglar temizlenmiş
```

### 3. Görselleştirme (bounding box'lı görsel)
- Her kelimenin etrafında renkli kutular
- Orijinal görsel üzerine çizilmiş

## 📁 Proje Yapısı

```
mtm-ocr-deepseek/
├── mtm_batch_ocr.py          # Ana batch OCR işleyici
├── web_ui.py                  # Flask web uygulaması
├── templates/
│   └── index.html             # Web arayüzü
├── DeepSeek-OCR/              # DeepSeek-OCR kodu
│   └── DeepSeek-OCR-master/
│       └── DeepSeek-OCR-vllm/
├── Dockerfile                 # Docker image tanımı
├── docker-compose.yml         # Docker Compose config
├── docker-entrypoint.sh       # Container başlangıç scripti
├── requirements.txt           # Python bağımlılıkları
├── README.md                  # Bu dosya
├── output/                    # Çıktı klasörü
│   ├── results/               # JSON ve TXT sonuçlar
│   ├── visualizations/        # Görselleştirilmiş resimler
│   └── images/                # Çıkarılan resim parçaları
└── uploads/                   # Yüklenen dosyalar
```

## ⚙️ Konfigürasyon

### Performans Ayarları

```python
# mtm_batch_ocr.py içinde veya komut satırından

# GPU memory kullanımı (0.0-1.0)
gpu_memory_utilization = 0.9

# Eşzamanlı işlem sayısı
max_concurrency = 50  # GPU belleğine göre ayarlayın

# Paralel görsel hazırlama
num_workers = 32  # CPU çekirdek sayısına göre
```

### OCR Modu Ayarları

DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py:

```python
# Çözünürlük modları:
# Tiny: BASE_SIZE=512, IMAGE_SIZE=512, CROP_MODE=False
# Small: BASE_SIZE=640, IMAGE_SIZE=640, CROP_MODE=False  
# Base: BASE_SIZE=1024, IMAGE_SIZE=1024, CROP_MODE=False
# Large: BASE_SIZE=1280, IMAGE_SIZE=1280, CROP_MODE=False
# Gundam: BASE_SIZE=1024, IMAGE_SIZE=640, CROP_MODE=True (ÖNERİLEN)

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6  # Düşük GPU belleği için 6, yüksek için 9
```

## 🔧 Sorun Giderme

### GPU Bellek Hatası (CUDA OOM)

```python
# config.py içinde düşürün:
MAX_CROPS = 4  # veya 3
max_concurrency = 20  # veya daha düşük
gpu_memory_utilization = 0.7
```

### vLLM Import Hatası

```bash
# VLLM ortam değişkenini ayarlayın
export VLLM_USE_V1=0
```

### Model İndirme Sorunları

```bash
# HuggingFace cache dizinini temizleyin
rm -rf ~/.cache/huggingface/hub

# Veya manuel indirin
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
```

### Docker Container GPU Erişemiyor

```bash
# nvidia-docker runtime'ı kontrol edin
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# Eğer hata alırsanız, nvidia-docker2'yi yeniden kurun
```

## 📊 Performans

Test ortamı: NVIDIA A100 40GB

| Gazete Sayısı | İşlem Süresi | Ortalama/Sayfa |
|---------------|--------------|----------------|
| 1             | ~8 saniye    | 8s             |
| 10            | ~35 saniye   | 3.5s           |
| 50            | ~2.5 dakika  | 3s             |
| 100           | ~5 dakika    | 3s             |

* Dinamik kırpma modu (Gundam) ile
* MAX_CONCURRENCY=50 ile

## 🤝 Katkıda Bulunma

Bu proje Medya Takip Merkezi için özel olarak geliştirilmiştir.

## 📄 Lisans

Bu proje DeepSeek-OCR'nin lisansına tabidir.

## 🙏 Teşekkürler

- [DeepSeek-AI](https://github.com/deepseek-ai) - DeepSeek-OCR modeli
- [vLLM](https://github.com/vllm-project/vllm) - Hızlı inference
- [Vary](https://github.com/Ucas-HaoranWei/Vary/) - Vision encoder
- [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/) - OCR altyapısı

## 📞 İletişim

Medya Takip Merkezi
Proje: MTM OCR Sistemi

---

**Not**: İlk çalıştırmada model indirileceği için internet bağlantısı gereklidir. Model boyutu yaklaşık 15GB'dır.

**GPU Gereksinimi**: En az 24GB VRAM önerilir. Daha düşük VRAM için MAX_CROPS ve max_concurrency değerlerini düşürün.

