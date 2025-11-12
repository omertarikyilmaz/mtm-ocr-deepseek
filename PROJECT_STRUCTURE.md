# MTM OCR - Proje Yapısı

## 📁 Dizin Yapısı

```
mtm-ocr-deepseek/
├── app/                          # Ana uygulama
│   ├── core/                     # OCR işleme motoru
│   │   ├── __init__.py
│   │   └── ocr_processor.py      # DeepSeek OCR ile işleme
│   │
│   ├── web/                      # Web arayüzü
│   │   ├── routes.py             # Flask API endpoints
│   │   ├── templates/
│   │   │   └── index.html        # Ana HTML (152 satır, temiz)
│   │   └── static/               # Static dosyalar
│   │       ├── css/
│   │       │   └── style.css     # Tüm stiller
│   │       └── js/
│   │           ├── common.js     # Ortak fonksiyonlar
│   │           ├── ocr.js        # OCR modülü
│   │           └── search.js     # Kelime arama modülü
│   │
│   └── utils/                    # Yardımcı araçlar
│
├── DeepSeek-OCR/                 # DeepSeek-OCR kütüphanesi
│   └── DeepSeek-OCR-master/
│       └── DeepSeek-OCR-vllm/
│
├── docker/                       # Docker yapılandırması
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-entrypoint.sh
│
├── run.py                        # Uygulama giriş noktası
├── requirements.txt              # Python bağımlılıkları
└── README.md                     # Proje dokümantasyonu
```

## 🔧 Modüller

### 1. OCR İşleme (`app/core/ocr_processor.py`)
**Görev:** Gazete görsellerini OCR ile işler
- DeepSeek-OCR modelini kullanır
- Batch işleme desteği
- Her kelimenin pozisyonunu çıkarır
- JSON formatında sonuç üretir

**Önemli:** Base64 görsel kaydı dahil

### 2. Web API (`app/web/routes.py`)
**Görev:** REST API endpoints
- `/upload` - Dosya yükleme
- `/process` - OCR işlemi
- `/results` - Sonuçları listele
- `/api/download/<id>` - JSON indir
- `/api/search/keywords` - Kelime arama

### 3. Frontend Modülleri

#### `ocr.js` - OCR İşleme Modülü
- Dosya yükleme (drag & drop)
- OCR işlemini başlatma
- İşlem durumunu izleme
- Sonuçları listeleme
- JSON indirme

#### `search.js` - Kelime Arama Modülü
- Virgülle ayrılmış kelime arama
- Görselde vurgulama (canvas)
- Her kelime farklı renk
- Vurgulu görsel indirme
- **Koordinat sistemi:** bbox (pixel) + scale

#### `common.js` - Ortak Fonksiyonlar
- Tab değiştirme
- Modal yönetimi

#### `style.css` - Tüm Stiller
- Modern, responsive tasarım
- Tab sistemi
- Upload zone
- Results grid
- Modal
- Canvas container

## 🚀 Çalıştırma

### Docker ile (Önerilen)
```bash
cd docker
docker-compose up -d
```

### Manuel
```bash
python run.py
```

**URL:** http://localhost:5000

## 📊 JSON Çıktı Formatı

```json
{
  "image_id": "uuid",
  "image_filename": "file.jpg",
  "timestamp": "20251112_154213",
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "word_count": 150,
  "words": [
    {
      "text": "kelime",
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 250,
        "y2": 230,
        "width": 150,
        "height": 30
      },
      "normalized_bbox": {
        "x1": 52,
        "y1": 104,
        "x2": 130,
        "y2": 120
      },
      "index": 0
    }
  ],
  "full_text": "OCR'dan çıkan tam metin",
  "raw_ocr_output": "Ham OCR çıktısı",
  "image_base64": "data:image/jpeg;base64,..."
}
```

## 🎨 Özellikler

### OCR İşleme
- ✅ Batch işleme
- ✅ Progress tracking
- ✅ Base64 görsel kaydı
- ✅ JSON export

### Kelime Arama
- ✅ Virgülle ayrılmış çoklu kelime
- ✅ Tüm gazetelerde arama
- ✅ Görselde vurgulama (kutu içinde)
- ✅ Her kelime farklı renk
- ✅ Vurgulu görsel indirme

## 🔍 Koordinat Sistemi

**DeepSeek OCR Çıktısı:**
- Normalize: `[x1, y1, x2, y2]` (0-999 arası)

**Backend İşleme:**
- Normalize → Pixel: `bbox` (orijinal görsel boyutunda)
- JSON'da hem `bbox` hem `normalized_bbox` var

**Frontend Vurgulama:**
```javascript
// Orijinal görsel boyutu
const originalWidth = result.image_size.width;
const originalHeight = result.image_size.height;

// Canvas scale
const scale = Math.min(1, maxWidth / originalWidth);

// Koordinat çevirisi
const x = bbox.x1 * scale;
const y = bbox.y1 * scale;
const w = (bbox.x2 - bbox.x1) * scale;
const h = (bbox.y2 - bbox.y1) * scale;
```

## 📝 Notlar

- HTML: 1289 → 152 satır (91% azaltma)
- Modüler yapı: Her modül bağımsız
- Test edilebilir: Her modül ayrı test edilebilir
- Bakım kolaylığı: Değişiklikler lokalize
- Performans: Static dosyalar cache'lenebilir
