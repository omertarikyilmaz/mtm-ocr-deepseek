#!/bin/bash

# MTM OCR - Başlat
# Servisleri başlatır

set -e

echo "========================================"
echo "MTM OCR - Başlatılıyor..."
echo "========================================"
echo ""

# Gerekli klasörleri oluştur
echo "[1/3] Klasörler oluşturuluyor..."
mkdir -p uploads output/results models

# Docker Compose ile başlat
echo "[2/3] Docker servisleri başlatılıyor..."
docker-compose up -d

# Durum kontrolü
echo "[3/3] Servis durumları kontrol ediliyor..."
sleep 5
docker-compose ps

echo ""
echo "========================================"
echo "✅ MTM OCR başlatıldı!"
echo "========================================"
echo ""
echo "🌐 Web Arayüzü: http://localhost"
echo "🔌 Backend API: http://localhost:5000"
echo "🤖 DeepSeek OCR: http://localhost:8000"
echo ""
echo "📊 Logları görüntülemek için:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Durdurmak için:"
echo "   ./stop.sh"
echo ""

