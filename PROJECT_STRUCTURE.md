# Proje Yapısı

```
mtm-ocr-deepseek/
├── app/                          # Ana uygulama kodu
│   ├── __init__.py              # Uygulama paketi
│   ├── core/                    # OCR işleme modülü
│   │   ├── __init__.py
│   │   └── ocr_processor.py     # MTMOCRProcessor (eski mtm_batch_ocr.py)
│   ├── web/                     # Web arayüzü
│   │   ├── __init__.py
│   │   ├── routes.py            # Flask routes (eski web_ui.py)
│   │   └── templates/           # HTML şablonları
│   │       └── index.html
│   └── utils/                   # Yardımcı fonksiyonlar
│       └── __init__.py
├── docker/                      # Docker yapılandırması
│   ├── Dockerfile               # Docker image tanımı
│   ├── docker-compose.yml      # Docker Compose yapılandırması
│   └── docker-entrypoint.sh    # Container başlangıç scripti
├── DeepSeek-OCR/               # DeepSeek-OCR kütüphanesi
│   └── DeepSeek-OCR-master/
│       └── DeepSeek-OCR-vllm/
├── run.py                       # Uygulama giriş noktası
├── requirements.txt             # Python bağımlılıkları
├── README.md                    # Proje dokümantasyonu
├── .gitignore                  # Git ignore kuralları
└── PROJECT_STRUCTURE.md        # Bu dosya
```

## Modüller

### `app/core/`
OCR işleme mantığı. DeepSeek-OCR modelini kullanarak görselleri işler.

### `app/web/`
Flask web uygulaması. REST API endpoint'leri ve HTML arayüzü.

### `docker/`
Docker container yapılandırması. GPU desteği ile çalışır.

## Çalıştırma

### Docker ile (Önerilen)
```bash
cd docker
docker-compose up -d
```

### Manuel
```bash
python run.py
```

