"""
MTM OCR Web UI
Basit web arayüzü ile gazete sayfalarını yükleyip OCR sonuçlarını görüntüleme
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import threading

# MTM Batch OCR'ı import et
from mtm_batch_ocr import MTMOCRProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max dosya boyutu

# Klasörleri oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global OCR processor (lazy loading)
ocr_processor = None
processing_status = {
    'is_processing': False,
    'current_file': '',
    'progress': 0,
    'total': 0,
    'status_message': ''
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_or_create_processor():
    """OCR processor'ı lazily oluştur"""
    global ocr_processor, processing_status
    if ocr_processor is None:
        try:
            processing_status['status_message'] = 'Model yukleniyor... (ilk calistirmada 5-10 dakika surebilir)'
            
            ocr_processor = MTMOCRProcessor(
                output_dir=app.config['OUTPUT_FOLDER'],
                max_concurrency=30
            )
            
            processing_status['status_message'] = 'Model hazir'
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"[ERROR] Model yukleme hatasi: {str(e)}"
            print(error_msg)
            print(error_details)
            processing_status['status_message'] = error_msg
            raise
            
    return ocr_processor

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Dosya yükleme"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    files = request.files.getlist('files[]')
    
    if not files:
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            # Benzersiz ID oluştur (UUID kullan)
            import uuid
            unique_id = str(uuid.uuid4())
            original_filename = secure_filename(file.filename)
            _, ext = os.path.splitext(original_filename)
            ext = ext.lower()  # .jpg, .jpeg, .png
            
            # Dosya adı: {unique_id}.{ext}
            unique_filename = f"{unique_id}{ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            uploaded_files.append({
                'id': unique_id,  # Benzersiz ID
                'filename': unique_filename,  # {id}.jpg
                'original_name': original_filename,
                'path': filepath
            })
    
    if not uploaded_files:
        return jsonify({'error': 'Geçerli dosya bulunamadı (sadece JPG, JPEG, PNG desteklenir)'}), 400
    
    return jsonify({
        'success': True,
        'files': uploaded_files,
        'count': len(uploaded_files)
    })

@app.route('/process', methods=['POST'])
def process_images():
    """Yüklenen görselleri işle"""
    global processing_status
    
    if processing_status['is_processing']:
        return jsonify({'error': 'Zaten bir işlem devam ediyor'}), 400
    
    data = request.get_json()
    filenames = data.get('filenames', [])
    
    if not filenames:
        return jsonify({'error': 'İşlenecek dosya bulunamadı'}), 400
    
    # Dosya yollarını oluştur
    image_paths = [
        os.path.join(app.config['UPLOAD_FOLDER'], filename)
        for filename in filenames
    ]
    
    # Var olmayan dosyaları filtrele
    image_paths = [p for p in image_paths if os.path.exists(p)]
    
    if not image_paths:
        return jsonify({'error': 'Geçerli dosya bulunamadı'}), 400
    
    # Background thread'de işle
    def process_background():
        global processing_status
        try:
            processing_status['is_processing'] = True
            processing_status['total'] = len(image_paths)
            processing_status['progress'] = 0
            processing_status['status_message'] = 'OCR islemi baslatiiliyor...'
            
            print(f"[INFO] {len(image_paths)} dosya islenecek")
            
            # Model yükleniyor (ilk çalıştırmada)
            processing_status['status_message'] = 'Model hazirlaniyor...'
            processor = get_or_create_processor()
            
            # İşleme başladı
            processing_status['status_message'] = f'OCR isleniyor... (0/{len(image_paths)})'
            print(f"[INFO] OCR islemi basladi: {len(image_paths)} dosya")
            
            # Progress callback fonksiyonu
            def update_progress(current, total, message):
                processing_status['progress'] = current
                processing_status['total'] = total
                processing_status['status_message'] = f'{message} ({current}/{total})'
                # Sadece önemli milestone'larda log yaz
                if current == 0 or current == total or current % max(1, total // 10) == 0:
                    print(f"[PROGRESS] {current}/{total} - {message}")
            
            results = processor.process_batch(
                image_paths,
                num_workers=16,
                progress_callback=update_progress
            )
            
            processing_status['status_message'] = f'Basarili: {len(results)} sayfa islendi'
            processing_status['progress'] = len(results)
            print(f"[SUCCESS] Islem tamamlandi: {len(results)} sonuc")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f'[ERROR] Hata: {str(e)}'
            processing_status['status_message'] = error_msg
            print(f"[ERROR] Islem hatasi:\n{error_details}")
        finally:
            processing_status['is_processing'] = False
            print("[INFO] Islem tamamlandi veya sonlandi")
    
    thread = threading.Thread(target=process_background)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'OCR işlemi başlatıldı'
    })

@app.route('/status')
def get_status():
    """İşlem durumunu kontrol et"""
    return jsonify(processing_status)

@app.route('/results')
def list_results():
    """İşlenmiş sonuçları listele"""
    results_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'results')
    
    if not os.path.exists(results_dir):
        return jsonify({'results': []})
    
    # JSON dosyalarını bul
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    json_files.sort(key=os.path.getmtime, reverse=True)  # En yeni önce
    
    results = []
    for json_file in json_files[:50]:  # Son 50 sonuç
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                results.append({
                    'id': Path(json_file).stem,
                    'filename': data['image_filename'],
                    'timestamp': data['timestamp'],
                    'word_count': data['word_count'],
                    'json_file': os.path.basename(json_file)
                })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    return jsonify({'results': results})

@app.route('/result/<result_id>')
def get_result(result_id):
    """Tek bir sonucu getir"""
    json_file = os.path.join(app.config['OUTPUT_FOLDER'], 'results', f'{result_id}.json')
    
    if not os.path.exists(json_file):
        return jsonify({'error': 'Sonuç bulunamadı'}), 404
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/download/<result_id>/<file_type>')
def download_result(result_id, file_type):
    """Sonuçları indir (sadece json)"""
    if file_type == 'json':
        directory = os.path.join(app.config['OUTPUT_FOLDER'], 'results')
        filename = f'{result_id}.json'
        if os.path.exists(os.path.join(directory, filename)):
            return send_from_directory(directory, filename, as_attachment=True)
        else:
            return "Dosya bulunamadı", 404
    else:
        return "Sadece JSON indirilebilir", 400

@app.route('/delete/<result_id>', methods=['DELETE'])
def delete_result(result_id):
    """Tek bir sonucu sil"""
    try:
        results_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'results')
        viz_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'visualizations')
        
        # JSON dosyasından bilgi al
        json_file = os.path.join(results_dir, f'{result_id}.json')
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # İlgili tüm dosyaları sil (TXT artık yok)
            files_to_delete = [
                os.path.join(results_dir, f'{result_id}.json'),
                os.path.join(viz_dir, f"{data['image_filename']}_{data['timestamp']}_boxes.jpg")
            ]
            
            deleted_count = 0
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            
            print(f"[INFO] Silindi: {result_id} ({deleted_count} dosya)")
            return jsonify({'success': True, 'message': f'{deleted_count} dosya silindi'})
        else:
            return jsonify({'error': 'Sonuc bulunamadi'}), 404
            
    except Exception as e:
        print(f"[ERROR] Silme hatasi: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete-all', methods=['DELETE'])
def delete_all_results():
    """Tüm sonuçları sil"""
    try:
        results_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'results')
        
        deleted_count = 0
        
        # Results klasöründeki tüm JSON dosyalarını sil
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(results_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_count += 1
        
        print(f"[INFO] Tum sonuclar silindi: {deleted_count} JSON dosyasi")
        return jsonify({'success': True, 'message': f'{deleted_count} JSON dosyasi silindi'})
        
    except Exception as e:
        print(f"[ERROR] Toplu silme hatasi: {e}")
        return jsonify({'error': str(e)}), 500

def main():
    """Web sunucusunu başlat"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MTM OCR Web UI')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host adresi')
    parser.add_argument('--port', type=int, default=5000, help='Port numarası')
    parser.add_argument('--debug', action='store_true', help='Debug modu')
    parser.add_argument('--preload-model', action='store_true', default=True, help='Model başlangıçta yüklensin')
    
    args = parser.parse_args()
    
    print(f"""
    ========================================================================
    MTM OCR - Web Arayuzu
    Medya Takip Merkezi
    ========================================================================
    
    URL: http://{args.host}:{args.port}
    Upload Directory: {app.config['UPLOAD_FOLDER']}
    Output Directory: {app.config['OUTPUT_FOLDER']}
    """)
    
    # Model'i başlangıçta yükle (kullanıcı beklemesin)
    if args.preload_model:
        print("\n" + "="*70)
        print("[INFO] MODEL ON YUKLEME BASLIYOR")
        print("       Container baslatildiginda model otomatik yuklenecek")
        print("       Ilk calistirmada: ~15GB model indirilecek (5-10 dakika)")
        print("="*70 + "\n")
        
        try:
            get_or_create_processor()
            
            print("\n" + "="*70)
            print("[SUCCESS] MODEL ON YUKLEME TAMAMLANDI")
            print("          Kullanicilar aninda OCR islemi yapabilir")
            print("="*70 + "\n")
            
        except Exception as e:
            import traceback
            print("\n" + "="*70)
            print(f"[ERROR] MODEL ON YUKLEME HATASI")
            print(f"        {e}")
            print(f"        Model ilk kulanimda yuklenecek")
            print("="*70 + "\n")
            print("Detayli hata:")
            print(traceback.format_exc())
    
    print(f"\n    Web Arayuzu: http://localhost:{args.port}\n")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == '__main__':
    main()

