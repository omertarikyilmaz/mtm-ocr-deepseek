"""
MTM Batch OCR Processor
Medya Takip Merkezi için birden fazla gazete sayfasını batch olarak okuyup 
her kelimenin pozisyonunu kaydeden sistem
"""

import os
import re
import json
import glob
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'

from PIL import Image, ImageDraw
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
        self.crop_mode = crop_mode
        
        # Output dizinlerini oluştur
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        
        # LLM'i başlat
        print("\n" + "="*70)
        print("🚀 DEEPSEEK OCR MODEL YÜKLENİYOR...")
        print("="*70)
        print(f"📦 Model: {model_path}")
        print(f"💾 GPU Memory: 90% kullanılacak")
        print(f"⚡ Max Concurrent: {max_concurrency}")
        print("\n📥 Model dosyaları indiriliyor/yükleniyor...")
        print("   (İlk çalıştırmada ~15GB model indirilecek, 5-10 dakika sürebilir)")
        print("   (Sonraki çalıştırmalarda cache'den yüklenecek, 30-60 saniye)\n")
        
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
        
        print("\n🔧 Model GPU'ya yüklendi, parametreler ayarlanıyor...")
        
        # Sampling parametreleri
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
        
        print("🖼️  Image processor hazırlanıyor...")
        self.processor = DeepseekOCRProcessor()
        
        print("\n" + "="*70)
        print("✅ MODEL TAMAMEN HAZIR! OCR işlemleri yapılabilir.")
        print("="*70 + "\n")
    
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
        # <|ref|>text<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|> formatını bul
        pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        word_positions = []
        
        for idx, (word_text, coordinates_str) in enumerate(matches):
            try:
                # Koordinatları parse et
                coordinates = eval(coordinates_str)
                
                # Her bbox için kelime bilgisi kaydet
                for bbox in coordinates:
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        
                        # Normalize edilmiş koordinatları (0-999) gerçek piksel koordinatlarına çevir
                        pixel_x1 = int(x1 / 999 * image_width)
                        pixel_y1 = int(y1 / 999 * image_height)
                        pixel_x2 = int(x2 / 999 * image_width)
                        pixel_y2 = int(y2 / 999 * image_height)
                        
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
                print(f"⚠️ Koordinat parse hatası: {e}")
                continue
        
        return word_positions
    
    def extract_text_only(self, text: str) -> str:
        """
        OCR çıktısından sadece metni çıkart (pozisyon tagları olmadan)
        
        Args:
            text: OCR çıktı metni
            
        Returns:
            Temiz metin
        """
        # Tüm ref ve det taglarını temizle
        pattern = r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>'
        clean_text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Fazla boşlukları temizle
        clean_text = re.sub(r'\n\n+', '\n\n', clean_text)
        clean_text = re.sub(r' +', ' ', clean_text)
        
        return clean_text.strip()
    
    def visualize_word_positions(
        self,
        image: Image.Image,
        word_positions: List[Dict],
        output_path: str
    ) -> Image.Image:
        """
        Kelime pozisyonlarını görsel üzerine çiz
        
        Args:
            image: Orijinal görsel
            word_positions: Kelime pozisyon bilgileri
            output_path: Çıktı dosya yolu
            
        Returns:
            Bounding box'lı görsel
        """
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Yarı saydam overlay
        overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)
        
        for word_info in word_positions:
            try:
                bbox = word_info['bbox']
                text = word_info['text']
                
                # Rastgele renk
                color = (
                    np.random.randint(50, 200),
                    np.random.randint(50, 200),
                    np.random.randint(50, 200)
                )
                color_alpha = color + (30,)
                
                # Bounding box çiz
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw2.rectangle([x1, y1, x2, y2], fill=color_alpha)
                
            except Exception as e:
                print(f"⚠️ Çizim hatası: {e}")
                continue
        
        img_draw.paste(overlay, (0, 0), overlay)
        img_draw.save(output_path)
        
        return img_draw
    
    def process_single_image(
        self,
        image_path: str,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown."
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
            print(f"❌ Görsel işleme hatası ({image_path}): {e}")
            return None
    
    def process_batch(
        self,
        image_paths: List[str],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        num_workers: int = 32,
        progress_callback: Optional[callable] = None
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
        print(f"\n📰 {len(image_paths)} gazete sayfası işleniyor...")
        
        if progress_callback:
            progress_callback(0, len(image_paths), "Görseller hazırlanıyor...")
        
        # Görselleri paralel olarak hazırla
        print("🔄 Görseller hazırlanıyor...")
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
            print("❌ İşlenecek görsel bulunamadı!")
            return []
        
        # Batch inference
        print("🤖 OCR işlemi yapılıyor...")
        if progress_callback:
            progress_callback(0, len(processed_images), "OCR işlemi yapılıyor...")
            
        batch_inputs = [img['cache_item'] for img in processed_images]
        
        outputs_list = self.llm.generate(
            batch_inputs,
            sampling_params=self.sampling_params
        )
        
        # Sonuçları işle ve kaydet
        print("💾 Sonuçlar kaydediliyor...")
        if progress_callback:
            progress_callback(0, len(processed_images), "Sonuçlar kaydediliyor...")
            
        results = []
        
        for idx, (output, img_data) in enumerate(zip(outputs_list, processed_images)):
            if progress_callback:
                progress_callback(idx + 1, len(processed_images), f"Sonuç kaydediliyor... ({idx+1}/{len(processed_images)})")
            try:
                # OCR çıktısı
                ocr_text = output.outputs[0].text
                
                # Dosya adı
                image_filename = Path(img_data['image_path']).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Kelime pozisyonlarını çıkart
                word_positions = self.extract_word_positions(
                    ocr_text,
                    img_data['width'],
                    img_data['height']
                )
                
                # Temiz metni çıkart
                clean_text = self.extract_text_only(ocr_text)
                
                # JSON sonuç
                result_data = {
                    'image_path': img_data['image_path'],
                    'image_filename': image_filename,
                    'timestamp': timestamp,
                    'image_size': {
                        'width': img_data['width'],
                        'height': img_data['height']
                    },
                    'word_count': len(word_positions),
                    'words': word_positions,
                    'full_text': clean_text,
                    'raw_ocr_output': ocr_text
                }
                
                # JSON dosyasını kaydet
                json_path = os.path.join(
                    self.output_dir,
                    'results',
                    f'{image_filename}_{timestamp}.json'
                )
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                # Temiz metni kaydet
                text_path = os.path.join(
                    self.output_dir,
                    'results',
                    f'{image_filename}_{timestamp}.txt'
                )
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                
                # Görselleştirme
                viz_path = os.path.join(
                    self.output_dir,
                    'visualizations',
                    f'{image_filename}_{timestamp}_boxes.jpg'
                )
                self.visualize_word_positions(
                    img_data['image'],
                    word_positions,
                    viz_path
                )
                
                results.append(result_data)
                
                print(f"✅ {image_filename}: {len(word_positions)} kelime bulundu")
                
            except Exception as e:
                print(f"❌ Sonuç kaydetme hatası ({img_data['image_path']}): {e}")
                continue
        
        print(f"\n🎉 İşlem tamamlandı! {len(results)} gazete başarıyla işlendi.")
        print(f"📁 Sonuçlar: {self.output_dir}/results/")
        print(f"🖼️  Görselleştirmeler: {self.output_dir}/visualizations/")
        
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
        print(f"❌ {args.input} konumunda görsel bulunamadı!")
        return
    
    print(f"📸 {len(image_paths)} görsel bulundu")
    
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

