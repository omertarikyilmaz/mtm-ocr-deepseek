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
    global ocr_processor
    if ocr_processor is None:
        print("🚀 OCR Processor başlatılıyor...")
        ocr_processor = MTMOCRProcessor(
            output_dir=app.config['OUTPUT_FOLDER'],
            max_concurrency=30
        )
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
            filename = secure_filename(file.filename)
            # Benzersiz dosya adı oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{timestamp}{ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            uploaded_files.append({
                'filename': unique_filename,
                'original_name': filename,
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
            processing_status['status_message'] = 'OCR işlemi başladı...'
            
            processor = get_or_create_processor()
            
            # İşleme
            results = processor.process_batch(
                image_paths,
                num_workers=16
            )
            
            processing_status['status_message'] = f'✅ {len(results)} sayfa başarıyla işlendi!'
            processing_status['progress'] = len(results)
            
        except Exception as e:
            processing_status['status_message'] = f'❌ Hata: {str(e)}'
        finally:
            processing_status['is_processing'] = False
    
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
                
                # Görselleştirme dosyasını bul
                viz_filename = f"{data['image_filename']}_{data['timestamp']}_boxes.jpg"
                viz_path = os.path.join(app.config['OUTPUT_FOLDER'], 'visualizations', viz_filename)
                
                results.append({
                    'id': Path(json_file).stem,
                    'filename': data['image_filename'],
                    'timestamp': data['timestamp'],
                    'word_count': data['word_count'],
                    'has_visualization': os.path.exists(viz_path),
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
        
        # Görselleştirme URL'i
        viz_filename = f"{data['image_filename']}_{data['timestamp']}_boxes.jpg"
        viz_path = os.path.join(app.config['OUTPUT_FOLDER'], 'visualizations', viz_filename)
        
        data['visualization_url'] = f"/visualization/{viz_filename}" if os.path.exists(viz_path) else None
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualization/<filename>')
def serve_visualization(filename):
    """Görselleştirme dosyasını servis et"""
    viz_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'visualizations')
    return send_from_directory(viz_dir, filename)

@app.route('/download/<result_id>/<file_type>')
def download_result(result_id, file_type):
    """Sonuçları indir (json, txt, veya image)"""
    if file_type == 'json':
        directory = os.path.join(app.config['OUTPUT_FOLDER'], 'results')
        filename = f'{result_id}.json'
    elif file_type == 'txt':
        directory = os.path.join(app.config['OUTPUT_FOLDER'], 'results')
        filename = f'{result_id}.txt'
    elif file_type == 'image':
        # JSON'dan bilgi al
        json_file = os.path.join(app.config['OUTPUT_FOLDER'], 'results', f'{result_id}.json')
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            directory = os.path.join(app.config['OUTPUT_FOLDER'], 'visualizations')
            filename = f"{data['image_filename']}_{data['timestamp']}_boxes.jpg"
        else:
            return "Dosya bulunamadı", 404
    else:
        return "Geçersiz dosya tipi", 400
    
    if os.path.exists(os.path.join(directory, filename)):
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        return "Dosya bulunamadı", 404

def main():
    """Web sunucusunu başlat"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MTM OCR Web UI')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host adresi')
    parser.add_argument('--port', type=int, default=5000, help='Port numarası')
    parser.add_argument('--debug', action='store_true', help='Debug modu')
    
    args = parser.parse_args()
    
    print(f"""
    ╔══════════════════════════════════════════╗
    ║   MTM OCR - Web Arayüzü                 ║
    ║   Medya Takip Merkezi                   ║
    ╚══════════════════════════════════════════╝
    
    🌐 URL: http://{args.host}:{args.port}
    📁 Upload: {app.config['UPLOAD_FOLDER']}
    📁 Output: {app.config['OUTPUT_FOLDER']}
    
    Tarayıcınızda açın: http://localhost:{args.port}
    """)
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == '__main__':
    main()

