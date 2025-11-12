#!/bin/bash

cd "$(dirname "$0")"

echo "========================================================================"
echo "MTM OCR - Docker Container Baslatiliyor"
echo "========================================================================"
echo ""

if command -v docker-compose &> /dev/null; then
    echo "[INFO] docker-compose kullaniliyor"
    docker-compose up -d --build
elif docker compose version &> /dev/null; then
    echo "[INFO] docker compose kullaniliyor"
    docker compose up -d --build
else
    echo "[HATA] Docker Compose bulunamadi!"
    echo ""
    echo "Yuklemek icin:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install docker-compose"
    exit 1
fi

echo ""
echo "[INFO] Container baslatildi"
echo "       Loglar icin: docker-compose logs -f"
echo "       Durdurmak icin: docker-compose down"
echo ""

