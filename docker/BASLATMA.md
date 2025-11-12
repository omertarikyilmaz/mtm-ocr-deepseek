# Sunucuda Çalıştırma

## Docker Compose Komutları

### Eski Versiyon (docker-compose - tire ile):
```bash
cd /home/ower/Projects/mtm-ocr-deepseek/docker
docker-compose build
docker-compose up -d
docker-compose logs -f
```

### Yeni Versiyon (docker compose - boşluk ile):
```bash
cd /home/ower/Projects/mtm-ocr-deepseek/docker
docker compose build
docker compose up -d
docker compose logs -f
```

## Hızlı Başlatma

```bash
# 1. Proje dizinine git
cd /home/ower/Projects/mtm-ocr-deepseek/docker

# 2. Build ve başlat (ilk çalıştırmada)
docker-compose up -d --build
# veya
docker compose up -d --build

# 3. Logları takip et
docker-compose logs -f
# veya
docker compose logs -f

# 4. Durdur
docker-compose down
# veya
docker compose down
```

## Sorun Giderme

Eğer "command not found" hatası alıyorsanız:

1. Docker Compose'un yüklü olup olmadığını kontrol edin:
```bash
docker-compose --version
# veya
docker compose version
```

2. Yüklü değilse, yükleyin:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker-compose

# veya Docker Compose V2 (önerilen)
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

3. Kullanıcıyı docker grubuna ekleyin:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

