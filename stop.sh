#!/bin/bash

# MTM OCR - Durdur
# Servisleri durdurur

set -e

echo "========================================"
echo "MTM OCR - Durduruluyor..."
echo "========================================"
echo ""

# Docker Compose ile durdur
docker-compose down

echo ""
echo "========================================"
echo "✅ MTM OCR durduruldu!"
echo "========================================"
echo ""

