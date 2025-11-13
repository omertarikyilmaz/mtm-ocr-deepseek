"""
DeepSeek OCR Servisi - Basit OCR (bbox mantığı kaldırılmış)
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from PIL import Image

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# Local imports
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


class SimpleOCRProcessor:
    """Basit OCR İşleyici - Sadece metin çıkarma"""
    
    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        output_dir: str = "/app/output",
        device: str = "0",
        max_concurrency: int = 30
    ):
        """
        Args:
            model_path: DeepSeek-OCR model yolu
            output_dir: Çıktı klasörü
            device: GPU device ID
            max_concurrency: Maksimum eşzamanlı işlem sayısı
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        
        self.model_path = model_path
        self.output_dir = output_dir
        
        # Output dizinlerini oluştur
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        
        print("\n[INFO] DeepSeek OCR modeli yukleniyor...")
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
    
    def process_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Görselleri OCR ile işle - Sadece metin çıkar
        
        Args:
            image_paths: Görsel dosya yolları
            
        Returns:
            İşlem sonuçları
        """
        print(f"\n[INFO] {len(image_paths)} gorsel isleniyor...")
        
        # Görselleri hazırla
        processed_images = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                image_width, image_height = image.size
                
                # Basit OCR prompt (bbox YOK)
                prompt = "<image>\nFree OCR."
                
                cache_item = {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": self.processor.tokenize_with_images(
                            images=[image],
                            bos=True,
                            eos=True,
                            cropping=True
                        )
                    },
                }
                
                processed_images.append({
                    'cache_item': cache_item,
                    'image_path': image_path,
                    'width': image_width,
                    'height': image_height
                })
            except Exception as e:
                print(f"[HATA] Gorsel isleme hatasi: {image_path} - {e}")
                continue
        
        if not processed_images:
            print("[HATA] Islenecek gorsel bulunamadi")
            return []
        
        # Batch OCR
        print(f"[INFO] OCR isleniyor: {len(processed_images)} gorsel")
        batch_inputs = [img['cache_item'] for img in processed_images]
        
        outputs_list = self.llm.generate(
            batch_inputs,
            sampling_params=self.sampling_params
        )
        
        # Sonuçları kaydet
        print(f"[INFO] Sonuclar kaydediliyor...")
        results = []
        
        for idx, (output, img_data) in enumerate(zip(outputs_list, processed_images)):
            try:
                # OCR metni
                ocr_text = output.outputs[0].text
                
                # Temiz metin (tag'leri kaldır)
                clean_text = self._clean_text(ocr_text)
                
                # Dosya bilgileri
                image_path_obj = Path(img_data['image_path'])
                image_filename = image_path_obj.name
                image_id = image_path_obj.stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Görseli base64'e çevir
                image_base64 = None
                try:
                    with open(img_data['image_path'], 'rb') as img_file:
                        image_data = img_file.read()
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                except Exception as e:
                    print(f"[HATA] Base64 cevrimi hatasi: {e}")
                
                # JSON sonuç
                result_data = {
                    'image_id': image_id,
                    'image_filename': image_filename,
                    'image_path': img_data['image_path'],
                    'timestamp': timestamp,
                    'image_size': {
                        'width': img_data['width'],
                        'height': img_data['height']
                    },
                    'text': clean_text,
                    'raw_ocr_output': ocr_text,
                    'image_base64': image_base64
                }
                
                # JSON dosyasını kaydet
                json_filename = f'{image_id}.json'
                json_path = os.path.join(self.output_dir, 'results', json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                results.append(result_data)
                print(f"[INFO] {image_filename}: {len(clean_text)} karakter islendi")
                
            except Exception as e:
                print(f"[HATA] Sonuc kaydetme hatasi: {e}")
                continue
        
        print(f"\n[INFO] Islem tamamlandi: {len(results)} dosya islendi")
        return results
    
    def _clean_text(self, text: str) -> str:
        """OCR çıktısından temiz metin çıkar"""
        import re
        
        # Ref tag'leri varsa onları çıkar
        pattern = r'<\|ref\|>(.*?)<\|/ref\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            clean_text = ' '.join(match.strip() for match in matches if match.strip())
        else:
            # Tag yok, direkt temizle
            clean_text = re.sub(r'<\|.*?\|>', '', text)
            clean_text = re.sub(r'\n\n+', '\n\n', clean_text)
            clean_text = re.sub(r' +', ' ', clean_text)
        
        return clean_text.strip()

