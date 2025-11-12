# 🚀 HIZLI BAŞLANGIÇ - MTM OCR

## 1️⃣ TEK KOMUTLA BAŞLAT (Docker)

```bash
# Projeye git
cd /home/ower/Projects/mtm-ocr-deepseek

# Container'ı başlat (ilk çalıştırma 10-15 dk)
docker-compose up -d

# Logları takip et
docker-compose logs -f
```

**Tarayıcıda aç**: http://localhost:5000

## 2️⃣ KULLANIM

1. **Gazete Yükle**: Web arayüzünde dosyaları sürükle-bırak
2. **İşle**: "OCR İşlemini Başlat" butonuna tıkla
3. **Sonuçları İndir**: JSON, TXT veya görsel olarak

## 3️⃣ ÇIKTILAR

Her gazete için 3 dosya:

- **JSON**: Tüm kelimeler + pozisyonlar
- **TXT**: Sadece temiz metin
- **JPG**: Görsel üzerinde bounding box'lar

```
output/
├── results/
│   ├── gazete1_20250112_143022.json
│   └── gazete1_20250112_143022.txt
└── visualizations/
    └── gazete1_20250112_143022_boxes.jpg
```

## 4️⃣ JSON FORMATI

```json
{
  "words": [
    {
      "text": "Başlık",
      "bbox": {
        "x1": 245,
        "y1": 120,
        "x2": 380,
        "y2": 155
      }
    }
  ]
}
```

## 5️⃣ KOMUT SATIRI

```bash
# Manuel işleme
python mtm_batch_ocr.py --input ./gazeteler/ --output ./sonuclar/
```

## 6️⃣ DOCKER KOMUTLARI

```bash
# Durdur
docker-compose down

# Yeniden başlat
docker-compose restart

# Logları gör
docker-compose logs -f
```

## ❓ SORUN GİDERME

### GPU Bellek Hatası
```python
# DeepSeek-OCR/.../config.py içinde:
MAX_CROPS = 4  # 6'dan düşür
```

### Container GPU Görmüyor
```bash
# Test et
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# Nvidia-docker yükle
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 📊 PERFORMANS

- **1 gazete**: ~8 saniye
- **10 gazete**: ~35 saniye  
- **100 gazete**: ~5 dakika

*(NVIDIA A100 40GB ile)*

## 📞 İHTİYAÇ

- **GPU**: NVIDIA (en az 24GB VRAM)
- **VRAM**: 24GB+ (A100, RTX 3090, RTX 4090, vb.)
- **Docker**: nvidia-docker runtime

---

**İLK ÇALIŞTIRMA**: Model otomatik indirilir (~15GB), 10-15 dk sürer

