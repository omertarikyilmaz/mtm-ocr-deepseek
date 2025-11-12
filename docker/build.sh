#!/bin/bash

cd "$(dirname "$0")"

echo "========================================================================"
echo "MTM OCR - Docker Image Build"
echo "========================================================================"
echo ""

if command -v docker-compose &> /dev/null; then
    echo "[INFO] docker-compose kullaniliyor"
    docker-compose build
elif docker compose version &> /dev/null; then
    echo "[INFO] docker compose kullaniliyor"
    docker compose build
else
    echo "[HATA] Docker Compose bulunamadi!"
    exit 1
fi

echo ""
echo "[INFO] Build tamamlandi"
echo ""

