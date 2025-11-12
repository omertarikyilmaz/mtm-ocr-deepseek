"""
MTM Batch OCR Processor
Medya Takip Merkezi için birden fazla gazete sayfasını batch olarak okuyup 
her kelimenin pozisyonunu kaydeden sistem
"""

import os
import re
import json
import glob
import base64
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# Import local modules
from deepseek_vllm.deepseek_ocr import DeepseekOCRForCausalLM
from deepseek_vllm.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from deepseek_vllm.process.image_process import DeepseekOCRProcessor

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


class MTMOCRProcessor:
    """Medya Takip Merkezi için OCR işleyici"""
    
    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        output_dir: str = "output",
        device: str = "0",
        max_concurrency: int = 50,
        crop_mode: bool = True
    ):
        """
        Args:
            model_path: DeepSeek-OCR model yolu
            output_dir: Çıktı klasörü
            device: GPU device ID
            max_concurrency: Maksimum eşzamanlı işlem sayısı
            crop_mode: Dinamik kırpma modu
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        
        self.model_path = model_path
        self.output_dir = output_dir
        # Originals klasörünü oluştur (görselleri saklamak için)
        self.originals_dir = os.path.join(output_dir, 'originals')
        os.makedirs(self.originals_dir, exist_ok=True)
        self.crop_mode = crop_mode
        
        # Output dizinlerini oluştur
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        
        print("\n[INFO] DeepSeek OCR modeli yukleniyor")
        print(f"       Model: {model_path}")
        
        import time
        start_time = time.time()
        
        self.llm = LLM(
            model=model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            enforce_eager=False,
            trust_remote_code=True,
            max_model_len=8192,
            swap_space=0,
            max_num_seqs=max_concurrency,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=40, 
                window_size=90, 
                whitelist_token_ids={128821, 128822}
            )
        ]
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )
        
        self.processor = DeepseekOCRProcessor()
        
        total_time = time.time() - start_time
        print(f"[INFO] Model yukleme tamamlandi ({total_time:.1f} saniye)")
    
    def extract_word_positions(
        self, 
        text: str, 
        image_width: int, 
        image_height: int
    ) -> List[Dict]:
        """
        OCR çıktısından kelimelerin pozisyonlarını çıkart
        
        Args:
            text: OCR çıktı metni
            image_width: Görsel genişliği
            image_height: Görsel yüksekliği
            
        Returns:
            Kelime pozisyon bilgileri listesi
        """
        word_positions = []
        
        # GROUNDING TAG OLMADAN: Modelin kendi formatını kontrol et
        # Bazen markdown format döndürür, bazen düz metin
        
        # DEBUG: Free OCR response'un başını göster
        print(f"[DEBUG FREE OCR] Response uzunluğu: {len(text)} karakter")
        print(f"[DEBUG FREE OCR] İlk 500 karakter: {text[:500]}")
        
        # Format 1: <|ref|>text<|/ref|><|det|>coords<|/det|>
        pattern1 = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
        matches1 = re.findall(pattern1, text, re.DOTALL)
        
        # Format 2: <|ref|>text<|/ref|><|box|>coords<|/box|>
        pattern2 = r'<\|ref\|>(.*?)<\|/ref\|><\|box\|>(.*?)<\|/box\|>'
        matches2 = re.findall(pattern2, text, re.DOTALL)
        
        # Hangisi daha fazla sonuç veriyorsa onu kullan
        if len(matches1) > 0:
            matches = matches1
            format_name = "det"
            print(f"[DEBUG FREE OCR] {len(matches1)} adet <|det|> match bulundu")
        elif len(matches2) > 0:
            matches = matches2
            format_name = "box"
            print(f"[DEBUG FREE OCR] {len(matches2)} adet <|box|> match bulundu")
        else:
            matches = []
            format_name = "none"
            print(f"[DEBUG FREE OCR] Hiç bbox match bulunamadı!")
        
        if format_name == "none":
            return []
        
        for idx, (word_text, coordinates_str) in enumerate(matches):
            try:
                # Koordinatları parse et (güvenli)
                import ast
                coordinates = ast.literal_eval(coordinates_str)
                
                # Liste içinde liste varsa düzleştir
                if isinstance(coordinates, list):
                    if len(coordinates) > 0 and isinstance(coordinates[0], list):
                        # [[x1,y1,x2,y2]] formatı
                        bbox_list = coordinates
                    else:
                        # [x1,y1,x2,y2] formatı
                        bbox_list = [coordinates]
                    
                    for bbox in bbox_list:
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = bbox[:4]
                            
                            # DEBUG: İlk 3 kelime için bbox'ları göster
                            if idx < 3:
                                print(f"[DEBUG FREE OCR] '{word_text}' normalize bbox: {bbox}")
                            
                            # DeepSeek OCR HER ZAMAN 0-999 arası normalize döndürür!
                            # Resmi kod: x1 = int(x1 / 999 * image_width)
                            # modeling_deepseekocr.py satır 104-108
                            pixel_x1 = int(x1 / 999 * image_width)
                            pixel_y1 = int(y1 / 999 * image_height)
                            pixel_x2 = int(x2 / 999 * image_width)
                            pixel_y2 = int(y2 / 999 * image_height)
                            
                            # DEBUG: İlk 3 kelime için pixel bbox'ları göster
                            if idx < 3:
                                print(f"[DEBUG FREE OCR] '{word_text}' pixel bbox: x1={pixel_x1}, y1={pixel_y1}, x2={pixel_x2}, y2={pixel_y2}, w={pixel_x2-pixel_x1}, h={pixel_y2-pixel_y1}")
                            
                            word_positions.append({
                                'text': word_text.strip(),
                                'bbox': {
                                    'x1': pixel_x1,
                                    'y1': pixel_y1,
                                    'x2': pixel_x2,
                                    'y2': pixel_y2,
                                    'width': pixel_x2 - pixel_x1,
                                    'height': pixel_y2 - pixel_y1
                                },
                                'normalized_bbox': {
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2
                                },
                                'index': idx
                            })
            except Exception as e:
                continue
        
        return word_positions
    
    def extract_text_only(self, text: str) -> str:
        """
        OCR çıktısından sadece metni çıkart
        """
        pattern = r'<\|ref\|>(.*?)<\|/ref\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            clean_text = ' '.join(match.strip() for match in matches if match.strip())
        else:
            clean_text = re.sub(r'<\|.*?\|>', '', text)
            clean_text = re.sub(r'\n\n+', '\n\n', clean_text)
            clean_text = re.sub(r' +', ' ', clean_text)
        
        return clean_text.strip()
    
    def visualize_word_positions(
        self,
        image_path: str,
        word_positions: List[Dict],
        output_path: str,
        show_text: bool = True,
        box_color: tuple = None,
        box_width: int = 2
    ) -> Image.Image:
        """
        JSON'dan alınan kelime pozisyonlarını görsel üzerinde göster
        DeepSeek'ten bağımsız, kendi kutu çizme sistemimiz
        
        Args:
            image_path: Orijinal görsel yolu
            word_positions: JSON'dan gelen kelime pozisyonları
            output_path: Çıktı dosya yolu
            show_text: Kutunun üstünde metni göster
            box_color: Kutu rengi (None ise rastgele)
            box_width: Kutu çizgi kalınlığı
            
        Returns:
            Bounding box'lı görsel
        """
        image = Image.open(image_path).convert('RGB')
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw, 'RGBA')
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        for idx, word_info in enumerate(word_positions):
            try:
                bbox = word_info['bbox']
                text = word_info.get('text', '')
                
                if box_color:
                    color = box_color
                else:
                    hue = (idx * 137.5) % 360
                    import colorsys
                    rgb = colorsys.hsv_to_rgb(hue/360, 0.7, 0.9)
                    color = tuple(int(c * 255) for c in rgb)
                
                x1, y1 = bbox['x1'], bbox['y1']
                x2, y2 = bbox['x2'], bbox['y2']
                draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
                
            except Exception as e:
                continue
        
        img_draw.save(output_path, quality=95)
        return img_draw
    
    def process_single_image(
        self,
        image_path: str,
        prompt: str = "<image>\n<|grounding|>\nFree OCR.",
        use_word_location: bool = True
    ) -> Dict:
        """
        Tek bir görseli işle
        
        Args:
            image_path: Görsel dosya yolu
            prompt: OCR prompt
            
        Returns:
            İşlem sonucu bilgileri
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size
            
            # Image processing
            cache_item = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": self.processor.tokenize_with_images(
                        images=[image],
                        bos=True,
                        eos=True,
                        cropping=self.crop_mode
                    )
                },
            }
            
            return {
                'cache_item': cache_item,
                'image': image,
                'image_path': image_path,
                'width': image_width,
                'height': image_height
            }
            
        except Exception as e:
            print(f"[HATA] Gorsel isleme hatasi: {image_path}")
            return None
    
    def locate_words_in_image(
        self,
        image_path: str,
        words: List[str],
        batch_size: int = 10
    ) -> Dict[str, Dict]:
        """
        Her kelime için Locate (REC) mode ile koordinat bul
        Prompt: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
        
        Bu mod Free OCR'dan DAHA DOĞRU pozisyon verir!
        """
        word_locations = {}
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size
        
        print(f"[LOCATE] {len(words)} kelime için pozisyon bulunuyor...")
        
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i+batch_size]
            
            for word in batch_words:
                try:
                    # ✅ FIX: <|grounding|> tag'ini ekle (yardımcı kaynaktan)
                    locate_prompt = f"<image>\n<|grounding|>\nLocate <|ref|>{word}<|/ref|> in the image."
                    
                    cache_item = {
                        "prompt": locate_prompt,
                        "multi_modal_data": {
                            "image": self.processor.tokenize_with_images(
                                images=[image],
                                bos=True,
                                eos=True,
                                cropping=self.crop_mode
                            )
                        },
                    }
                    
                    output = self.llm.generate([cache_item], sampling_params=self.sampling_params)[0]
                    response = output.outputs[0].text
                    
                    # DEBUG: İlk 3 kelime için response'u göster
                    if len(word_locations) < 3:
                        print(f"[DEBUG] '{word}' için DeepSeek response: {response[:300]}...")
                    
                    # ✅ FIX: Doğru regex pattern (ref ve det tag'lerini birlikte yakala)
                    # Pattern: <|ref|>kelime<|/ref|><|det|>[[coords]]<|/det|>
                    pattern = r'<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>\s*(?P<coords>\[.*?\])\s*<\|/det\|>'
                    coords_match = re.search(pattern, response, re.DOTALL)
                    
                    if coords_match:
                        coords_str = coords_match.group('coords').strip()
                        
                        # DEBUG: Raw coords string
                        if len(word_locations) < 3:
                            print(f"[DEBUG] '{word}' raw coords: {coords_str}")
                        
                        # ✅ FIX: ast.literal_eval kullan (daha güvenli)
                        import ast
                        parsed = ast.literal_eval(coords_str)
                        
                        # ✅ FIX: Çoklu bbox desteği (yardımcı kaynaktan)
                        # Tek bbox: [x1,y1,x2,y2] veya [[x1,y1,x2,y2]]
                        # Çoklu bbox: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
                        if isinstance(parsed, list) and len(parsed) == 4 and all(isinstance(n, (int, float)) for n in parsed):
                            # Tek bbox: [x1,y1,x2,y2]
                            box_coords = [parsed]
                            if len(word_locations) < 3:
                                print(f"[DEBUG] '{word}' tek bbox tespit edildi")
                        elif isinstance(parsed, list):
                            # Çoklu bbox veya [[x1,y1,x2,y2]]
                            box_coords = parsed
                            if len(word_locations) < 3:
                                print(f"[DEBUG] '{word}' {len(box_coords)} bbox tespit edildi")
                        else:
                            print(f"[UYARI] '{word}' için beklenmeyen format: {type(parsed)}")
                            continue
                        
                        # Her bbox için işle (aynı kelime birden fazla yerde olabilir!)
                        word_lower = word.lower()
                        word_locations[word_lower] = []  # Liste olarak sakla
                        
                        for idx, bbox in enumerate(box_coords):
                            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                # DEBUG
                                if len(word_locations) < 3:
                                    print(f"[DEBUG] '{word}' bbox {idx+1}: normalize {bbox}")
                                    print(f"[DEBUG] Görsel boyutu: {image_width}x{image_height}")
                                
                                # Normalize (0-999) → Pixel
                                x1 = int(float(bbox[0]) / 999 * image_width)
                                y1 = int(float(bbox[1]) / 999 * image_height)
                                x2 = int(float(bbox[2]) / 999 * image_width)
                                y2 = int(float(bbox[3]) / 999 * image_height)
                                
                                # DEBUG
                                if len(word_locations) < 3:
                                    print(f"[DEBUG] '{word}' bbox {idx+1}: pixel x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={x2-x1}, h={y2-y1}")
                                
                                word_locations[word_lower].append({
                                    'bbox': {
                                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                        'width': x2 - x1, 'height': y2 - y1
                                    }
                                })
                        
                        if len(word_locations) % 10 == 0:
                            print(f"[LOCATE] {len(word_locations)}/{len(words)} kelime bulundu")
                                    
                except Exception as e:
                    print(f"[LOCATE] Hata '{word}': {e}")
                    continue
        
        print(f"[LOCATE] Toplam {len(word_locations)} kelime pozisyonu bulundu")
        return word_locations
    
    def process_batch(
        self,
        image_paths: List[str],
        prompt: str = "<image>\n<|grounding|>\nFree OCR.",
        num_workers: int = 32,
        progress_callback: Optional[callable] = None,
        use_word_location: bool = False  # Free OCR bbox'larını kullan
    ) -> List[Dict]:
        """
        Birden fazla görseli batch olarak işle
        
        Args:
            image_paths: Görsel dosya yolları listesi
            prompt: OCR prompt
            num_workers: Paralel işlem sayısı
            progress_callback: Progress güncellemesi için callback fonksiyonu
            
        Returns:
            Tüm görseller için OCR sonuçları
        """
        print(f"\n[INFO] {len(image_paths)} gazete sayfasi isleniyor...")
        
        if progress_callback:
            progress_callback(0, len(image_paths), "Gorseller hazirlaniyor")
        
        # Görselleri paralel olarak hazırla
        print("[1/3] Gorseller hazirlaniyor...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            processed_images = list(
                executor.map(
                    lambda p: self.process_single_image(p, prompt),
                    image_paths
                )
            )
        
        # None değerleri filtrele
        processed_images = [img for img in processed_images if img is not None]
        
        if not processed_images:
            print("[HATA] Islenecek gorsel bulunamadi")
            return []
        if progress_callback:
            progress_callback(0, len(processed_images), "OCR islemi yapiliyor")
            
        batch_inputs = [img['cache_item'] for img in processed_images]
        
        outputs_list = self.llm.generate(
            batch_inputs,
            sampling_params=self.sampling_params
        )
        
        if progress_callback:
            progress_callback(0, len(processed_images), "Sonuclar kaydediliyor")
            
        results = []
        
        for idx, (output, img_data) in enumerate(zip(outputs_list, processed_images)):
            if progress_callback:
                progress_callback(idx + 1, len(processed_images), f"Sonuc kaydediliyor ({idx+1}/{len(processed_images)})")
            try:
                # OCR çıktısı
                ocr_text = output.outputs[0].text
                
                # Dosya adı (benzersiz ID'den)
                image_path_obj = Path(img_data['image_path'])
                image_filename = image_path_obj.name  # {unique_id}.jpg formatında
                image_id = image_path_obj.stem  # Benzersiz ID (uzantısız)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Yüklenen görsel yolu (uploads klasöründeki orijinal)
                original_upload_path = img_data['image_path']
                
                # Kelime pozisyonlarını çıkart
                word_positions = self.extract_word_positions(
                    ocr_text,
                    img_data['width'],
                    img_data['height']
                )
                
                # Temiz metni çıkart
                clean_text = self.extract_text_only(ocr_text)
                
                # Görseli base64 olarak oku - YUKLENEN GORSELI DIREKT AL
                image_base64 = None
                try:
                    # Yüklenen görseli direkt al (img_data['image_path'] = uploads/{unique_id}.jpg)
                    upload_image_path = img_data['image_path']
                    if os.path.exists(upload_image_path):
                        with open(upload_image_path, 'rb') as img_file:
                            image_data = img_file.read()
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                            print(f"[INFO] Gorsel base64'e cevrildi: {upload_image_path}")
                    else:
                        print(f"[HATA] Gorsel bulunamadi: {upload_image_path}")
                except Exception as e:
                    print(f"[HATA] Gorsel base64'e cevrilemedi: {e}")
                    import traceback
                    print(traceback.format_exc())
                
                # Free OCR genelde pozisyonları eksik verir
                # Her kelime için Locate (REC) mode kullan - DAHA DOĞRU
                print(f"[INFO] Free OCR'dan {len(word_positions)} kelime pozisyonu bulundu")
                
                if use_word_location:
                    # Tüm kelimeleri çıkar
                    all_words = clean_text.split()
                    all_words = [w.strip('.,!?;:()[]{}\"\'') for w in all_words if len(w.strip('.,!?;:()[]{}\"\'')) > 0]
                    
                    # Unique kelimeleri bul
                    seen = set()
                    unique_words = []
                    for word in all_words:
                        word_lower = word.lower()
                        if word_lower not in seen and len(word) > 1:  # Tek harfleri atla
                            seen.add(word_lower)
                            unique_words.append(word)
                    
                    print(f"[INFO] Locate mode ile {len(unique_words)} unique kelime aranacak...")
                    
                    # Her kelime için Locate mode ile pozisyon bul
                    word_locations = self.locate_words_in_image(
                        img_data['image_path'], 
                        unique_words, 
                        batch_size=10  # Daha küçük batch
                    )
                    
                    print(f"[INFO] Locate mode'dan {len(word_locations)} kelime pozisyonu bulundu")
                    
                    # DEBUG: İlk 3 kelimenin bbox'larını göster
                    if word_locations:
                        print(f"[DEBUG] İlk 3 kelime lokasyonu:")
                        for word, bbox_list in list(word_locations.items())[:3]:
                            print(f"  '{word}': {len(bbox_list)} bbox bulundu")
                            for i, bbox_data in enumerate(bbox_list[:2]):  # İlk 2 bbox'ı göster
                                print(f"    [{i+1}] {bbox_data['bbox']}")
                    else:
                        print(f"[UYARI] Locate mode hiç kelime bulamadı! Free OCR bbox'ları kullanılacak.")
                    
                    # ✅ FIX: Çoklu bbox desteği - her kelime tekrarı için sıradaki bbox'ı kullan
                    word_positions = []
                    bbox_usage_count = {}  # Her kelime için kaç bbox kullandığımızı takip et
                    
                    for idx, word_text in enumerate(all_words):
                        word_lower = word_text.lower()
                        if word_lower in word_locations:
                            # Bu kelimenin kaçıncı tekrarı?
                            used_count = bbox_usage_count.get(word_lower, 0)
                            bbox_list = word_locations[word_lower]
                            
                            # Eğer bu tekrar için bbox varsa kullan
                            if used_count < len(bbox_list):
                                word_positions.append({
                                    'text': word_text,
                                    'bbox': bbox_list[used_count]['bbox'],
                                    'index': idx,
                                    'occurrence': used_count + 1  # Kaçıncı tekrar olduğunu belirt
                                })
                                bbox_usage_count[word_lower] = used_count + 1
                            else:
                                # Bbox kalmadı, bu tekrarı atla veya uyarı ver
                                if used_count == len(bbox_list):  # İlk kez bbox bittiğinde uyar
                                    print(f"[UYARI] '{word_text}' için {len(bbox_list)} bbox var ama {used_count+1}. tekrar isteniyor")
                                bbox_usage_count[word_lower] = used_count + 1
                
                # JSON sonuç
                result_data = {
                    'image_id': image_id,
                    'image_filename': image_filename,
                    'image_path': original_upload_path,
                    'timestamp': timestamp,
                    'image_size': {
                        'width': img_data['width'],
                        'height': img_data['height']
                    },
                    'word_count': len(word_positions),
                    'words': word_positions,
                    'full_text': clean_text,
                    'raw_ocr_output': ocr_text,
                    'image_base64': image_base64
                }
                
                # Base64 kontrolü
                if image_base64 is None:
                    print(f"[HATA] image_base64 None! Gorsel: {original_upload_path}")
                else:
                    print(f"[INFO] image_base64 hazir, uzunluk: {len(image_base64)} karakter")
                
                # JSON dosyasını kaydet - DOSYA ADI = image_id.json (basit ve garantili)
                json_filename = f'{image_id}.json'
                json_path = os.path.join(
                    self.output_dir,
                    'results',
                    json_filename
                )
                print(f"[INFO] JSON kaydediliyor: {json_path}")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                # TXT dosyası artık kaydedilmiyor - her şey JSON'da
                
                # Görselleştirme YOK - Kullanıcı manuel seçecek
                # viz_path = os.path.join(
                #     self.output_dir,
                #     'visualizations',
                #     f'{image_filename}_{timestamp}_boxes.jpg'
                # )
                # self.visualize_word_positions(
                #     img_data['image_path'],  # Orijinal görsel yolu
                #     word_positions,
                #     viz_path,
                #     show_text=True,  # Kelimeyi göster
                #     box_width=2  # İnce çizgi
                # )
                
                results.append(result_data)
                
                print(f"[INFO] {image_filename}: {len(word_positions)} kelime islendi")
                
            except Exception as e:
                print(f"[HATA] Sonuc kaydetme hatasi: {img_data['image_path']}")
                continue
        
        print(f"\n[INFO] Islem tamamlandi: {len(results)} dosya islendi")
        print(f"       Sonuclar: {self.output_dir}/results/")
        
        return results


def main():
    """Ana fonksiyon - CLI kullanımı için"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MTM Batch OCR - Gazete sayfalarını toplu olarak OCR ile işle'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Girdi klasörü veya dosya pattern (örn: ./gazeteler/*.jpg)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Çıktı klasörü'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='deepseek-ai/DeepSeek-OCR',
        help='Model yolu'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='GPU device ID'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=50,
        help='Maksimum eşzamanlı işlem sayısı'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=32,
        help='Paralel işlem sayısı'
    )
    parser.add_argument(
        '--no-crop',
        action='store_true',
        help='Dinamik kırpma modunu devre dışı bırak'
    )
    
    args = parser.parse_args()
    
    # Görsel dosyalarını bul
    if os.path.isdir(args.input):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
    else:
        image_paths = glob.glob(args.input)
    
    if not image_paths:
        print(f"[HATA] {args.input} konumunda gorsel bulunamadi")
        return
    
    print(f"[INFO] {len(image_paths)} gorsel bulundu")
    
    # Processor'ı başlat
    processor = MTMOCRProcessor(
        model_path=args.model,
        output_dir=args.output,
        device=args.device,
        max_concurrency=args.concurrency,
        crop_mode=not args.no_crop
    )
    
    # Batch işleme
    results = processor.process_batch(
        image_paths,
        num_workers=args.workers
    )
    
    # Özet rapor
    if results:
        total_words = sum(r['word_count'] for r in results)
        print(f"\n📊 ÖZET RAPOR:")
        print(f"   - İşlenen sayfa: {len(results)}")
        print(f"   - Toplam kelime: {total_words}")
        print(f"   - Ortalama kelime/sayfa: {total_words/len(results):.1f}")


if __name__ == "__main__":
    main()

