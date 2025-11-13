#!/bin/bash

# MTM OCR - Güncelle
# GitHub'tan güncellemeleri çeker ve build eder (başlatmaz)

set -e

echo "========================================"
echo "MTM OCR - Güncelleme"
echo "========================================"
echo ""

# Git durumu
echo "[1/3] Git durumu kontrol ediliyor..."
git status

echo ""
read -p "Devam edilsin mi? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "İptal edildi."
    exit 1
fi

# Git pull
echo "[2/3] GitHub'tan güncellemeler alınıyor..."
git pull origin main

# Docker build (başlatmadan)
echo "[3/3] Docker imajları build ediliyor..."
docker compose build

echo ""
echo "========================================"
echo "✅ Güncelleme tamamlandı!"
echo "========================================"
echo ""
echo "Servisleri başlatmak için:"
echo "   ./start.sh"
echo ""

