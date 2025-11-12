# Bilinen Sorunlar / Known Issues

Bu dokümant, MTM OCR projesindeki bilinen sorunları ve çözüm çabalarını detaylı olarak açıklar.

---

## 🚨 KRİTİK SORUN: Kelime Pozisyon Koordinatları Hatalı

**Tarih:** 12 Kasım 2025  
**Durum:** 🔴 Çözülmedi - Aktif araştırma  
**Etki:** Yüksek - Pozisyon tabanlı özellikler kullanılamaz

### Problem Açıklaması

DeepSeek-OCR modelinden alınan kelime pozisyon koordinatları (bounding box - bbox) tutarsız ve yanlış değerler döndürmektedir.

#### Gözlemlenen Davranış

**Örnek JSON Çıktısı:**
```json
{
  "image_size": {
    "width": 331,
    "height": 437
  },
  "words": [
    {
      "text": "CHP'DE",
      "bbox": {
        "x1": 0,
        "y1": 1,
        "x2": 329,
        "y2": 436,
        "width": 329,
        "height": 435
      }
    },
    {
      "text": "İDDİANAME",
      "bbox": {
        "x1": 0,
        "y1": 1,
        "x2": 328,
        "y2": 434,
        "width": 328,
        "height": 433
      }
    }
  ]
}
```

**Sorunlar:**
1. **Tüm Sayfa Kaplama:** "CHP'DE" gibi tek bir kelime, neredeyse tüm sayfa yüksekliğini (%99) ve genişliğini (%99) kaplayacak şekilde bbox döndürüyor.
2. **x1=0, y1=1 Tekrarı:** Farklı kelimeler aynı başlangıç koordinatlarını alıyor.
3. **Gerçekçi Olmayan Boyutlar:** Tek bir kelimenin 329x435 piksel boyutunda olması mümkün değil.

### Beklenen Davranış

Bir kelime için beklenen bbox örneği:
```json
{
  "text": "CHP'DE",
  "bbox": {
    "x1": 45,
    "y1": 23,
    "x2": 112,
    "y2": 48,
    "width": 67,
    "height": 25
  }
}
```

### Denenen Çözümler

#### 1. ✅ Free OCR → Locate (REC) Mode Geçişi
**Tarih:** 12 Kasım 2025  
**Commit:** `04136f8`, `a6e0b66`

**Değişiklik:**
```python
# Önceki: Free OCR mode (hızlı ama hassasiyetsiz)
prompt = "<image>\nFree OCR."

# Sonrası: Her kelime için Locate (REC) mode
# 1. Free OCR ile metni oku
# 2. Her kelime için ayrı Locate prompt gönder
locate_prompt = f"<image>\nLocate <|ref|>{word}<|/ref|> in the image."
```

**Sonuç:** Yavaşladı ama pozisyon hatası devam etti ❌

#### 2. ✅ Koordinat Dönüşüm Düzeltmesi
**Tarih:** 12 Kasım 2025  
**Commit:** `cf3a60b`, `a6e0b66`

**Problem:** Normalize (0-999) vs Pixel koordinat karışıklığı

**Çözüm:** DeepSeek resmi kodunu (modeling_deepseekocr.py) inceleyerek doğru formülü uyguladık:
```python
# DeepSeek HER ZAMAN 0-999 arası normalize döndürür
# Resmi formül (modeling_deepseekocr.py satır 104-108):
x1 = int(norm_x1 / 999 * image_width)
y1 = int(norm_y1 / 999 * image_height)
x2 = int(norm_x2 / 999 * image_width)
y2 = int(norm_y2 / 999 * image_height)
```

**Sonuç:** Formül doğru ama pozisyon hatası devam ediyor ❌

#### 3. 🔄 Debug Log Sistemi Eklendi
**Tarih:** 12 Kasım 2025  
**Commit:** `1e98d3b`

**Eklenen Log'lar:**
```python
# 1. DeepSeek'in ham response'u
print(f"[DEBUG] '{word}' için DeepSeek response: {response[:200]}...")

# 2. Normalize bbox (0-999)
print(f"[DEBUG] '{word}' normalize bbox: {bbox}")

# 3. Görsel boyutu
print(f"[DEBUG] Görsel boyutu: {image_width}x{image_height}")

# 4. Hesaplanan pixel bbox
print(f"[DEBUG] '{word}' pixel bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

# 5. Locate mode bulgu sayısı
print(f"[INFO] Locate mode'dan {len(word_locations)} kelime pozisyonu bulundu")
```

**Sonuç:** Veri toplanıyor, analiz edilecek 🔄

### Şüpheli Nedenler

#### 1. Model Prompt Formatı Yanlış Olabilir
DeepSeek-OCR'ın farklı task'lar için farklı prompt formatları var:
- Free OCR: `<image>\nFree OCR.`
- Locate (REC): `<image>\nLocate <|ref|>xxx<|/ref|> in the image.`
- Ground (DET): `<image>\nGround <|grounding|>xxx<|/grounding|> in the image.`

**Olası Sorun:** Locate mode'u yanlış kullanıyor olabiliriz.

#### 2. Model Yanlış Yüklenmiş Olabilir
vLLM ile model yüklerken bir hata olmuş ve model düzgün çalışmıyor olabilir.

#### 3. Image Preprocessing Hatası
Görsel işleme (resize, crop) sırasında koordinat bilgisi bozuluyor olabilir.

#### 4. Model Inherent Limitation
DeepSeek-OCR modeli bu task için uygun olmayabilir.

### Yapılacaklar / TODO

#### Kısa Vadeli (Debug)
- [ ] Debug log'larından DeepSeek'in ham response'unu incele
- [ ] Normalize bbox değerlerini kontrol et (0-999 arası mı?)
- [ ] DeepSeek resmi demo ile aynı görseli test et
- [ ] vLLM yerine orijinal Transformers implementasyonu dene

#### Orta Vadeli (Alternatif Çözümler)
- [ ] PaddleOCR entegrasyonu değerlendir
- [ ] TrOCR + CRAFT Text Detection pipeline kur
- [ ] GOT-OCR 2.0 (Vary-toy tabanlı) dene
- [ ] Microsoft's TrOCR + LayoutLM kombinasyonu

#### Uzun Vadeli (Yeniden Tasarım)
- [ ] Özel OCR + Text Detection pipeline geliştur
- [ ] Fine-tune edilmiş model eğit (gazete sayfaları için)
- [ ] Manuel koordinat düzeltme UI ekle

### Geçici Çözüm / Workaround

**Şu anki kullanım:**
```python
# ✅ ÇALIŞIYOR: Sadece metin çıkarma
result = processor.process_batch(image_paths)
full_text = result[0]['full_text']

# ❌ ÇALIŞMIYOR: Pozisyon bilgisi
words = result[0]['words']
for word in words:
    bbox = word['bbox']  # Yanlış değerler!
```

**Öneriler:**
- Sadece metin çıkarma için kullan
- Pozisyon gerektiren uygulamalar için **bekle** veya **alternatif OCR kullan**

### Alternatif OCR Araçları

Eğer pozisyon bilgisi kritikse:

1. **PaddleOCR** (Önerilen)
   - ✅ Doğru bbox koordinatları
   - ✅ Türkçe desteği iyi
   - ✅ Hızlı
   ```bash
   pip install paddlepaddle paddleocr
   ```

2. **GOT-OCR 2.0**
   - ✅ Vary-toy tabanlı, görsel tanıma güçlü
   - ✅ Bounding box desteği
   - ⚠️ Daha yavaş

3. **CRAFT + TrOCR**
   - ✅ Text detection (CRAFT) + Recognition (TrOCR)
   - ✅ Çok doğru koordinatlar
   - ⚠️ İki aşamalı, daha karmaşık

### İletişim

Bu sorunla ilgili güncellemeler için:
- GitHub Issues: [Proje GitHub](https://github.com/omertarikyilmaz/mtm-ocr-deepseek)
- Debug log'ları: `KNOWN_ISSUES.md` dosyası güncellenecek

---

**Son Güncelleme:** 12 Kasım 2025  
**Durum:** 🔴 Çözülmedi - Aktif araştırma devam ediyor

