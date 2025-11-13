"""
MTM OCR Backend - Flask API
Basit OCR servisi - Sadece metin çıkarma
"""

import os
import json
import uuid
import base64
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
CORS(app)

# Konfigürasyon
UPLOAD_FOLDER = '/app/uploads'
OUTPUT_FOLDER = '/app/output'
DEEPSEEK_API_URL = 'http://deepseek:8000'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'results'), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================================
# UPLOAD ENDPOINT
# ============================================================================

@app.route('/api/upload', methods=['POST'])
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
            unique_id = str(uuid.uuid4())
            original_filename = secure_filename(file.filename)
            ext = os.path.splitext(original_filename)[1].lower()
            unique_filename = f"{unique_id}{ext}"
            
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(filepath)
            uploaded_files.append({
                'id': unique_id,
                'filename': unique_filename,
                'original_name': original_filename,
                'path': filepath
            })
    
    if not uploaded_files:
        return jsonify({'error': 'Geçerli dosya bulunamadı'}), 400
    
    return jsonify({
        'success': True,
        'files': uploaded_files,
        'count': len(uploaded_files)
    })

# ============================================================================
# OCR PROCESSING ENDPOINT
# ============================================================================

@app.route('/api/process', methods=['POST'])
def process_images():
    """OCR işleme - DeepSeek servisine yönlendir"""
    data = request.get_json()
    filenames = data.get('filenames', [])
    
    if not filenames:
        return jsonify({'error': 'İşlenecek dosya bulunamadı'}), 400
    
    # DeepSeek servisine yönlendir
    try:
        response = requests.post(
            f'{DEEPSEEK_API_URL}/api/ocr',
            json={'filenames': filenames},
            timeout=600  # 10 dakika timeout
        )
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'DeepSeek servisi hatası: {str(e)}'}), 500

# ============================================================================
# STATUS ENDPOINT
# ============================================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """İşlem durumu"""
    try:
        response = requests.get(f'{DEEPSEEK_API_URL}/api/status', timeout=5)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException:
        return jsonify({
            'is_processing': False,
            'status_message': 'DeepSeek servisi hazırlanıyor...'
        })

# ============================================================================
# RESULTS ENDPOINTS
# ============================================================================

@app.route('/api/results', methods=['GET'])
def list_results():
    """İşlenmiş sonuçları listele"""
    results_dir = os.path.join(OUTPUT_FOLDER, 'results')
    
    if not os.path.exists(results_dir):
        return jsonify({'results': []})
    
    json_files = []
    for f in os.listdir(results_dir):
        if f.endswith('.json'):
            json_files.append(os.path.join(results_dir, f))
    
    json_files.sort(key=os.path.getmtime, reverse=True)
    
    results = []
    for json_file in json_files[:50]:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result_id = Path(json_file).stem
            results.append({
                'id': result_id,
                'filename': data.get('image_filename', 'unknown'),
                'timestamp': data.get('timestamp', ''),
                'text_length': len(data.get('text', '')),
                'json_file': os.path.basename(json_file)
            })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    return jsonify({'results': results})

@app.route('/api/result/<result_id>', methods=['GET'])
def get_result(result_id):
    """Tek bir sonucu getir"""
    json_file = os.path.join(OUTPUT_FOLDER, 'results', f'{result_id}.json')
    
    if not os.path.exists(json_file):
        return jsonify({'error': 'Sonuç bulunamadı'}), 404
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<result_id>', methods=['GET'])
def download_result(result_id):
    """JSON dosyasını indir"""
    results_dir = os.path.join(OUTPUT_FOLDER, 'results')
    filename = f'{result_id}.json'
    return send_from_directory(results_dir, filename, as_attachment=True)

@app.route('/api/delete/<result_id>', methods=['DELETE'])
def delete_result(result_id):
    """Tek bir sonucu sil"""
    json_file = os.path.join(OUTPUT_FOLDER, 'results', f'{result_id}.json')
    
    if os.path.exists(json_file):
        os.remove(json_file)
        return jsonify({'success': True})
    
    return jsonify({'error': 'Sonuç bulunamadı'}), 404

@app.route('/api/delete-all', methods=['DELETE'])
def delete_all_results():
    """Tüm sonuçları sil"""
    results_dir = os.path.join(OUTPUT_FOLDER, 'results')
    deleted_count = 0
    
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                os.remove(os.path.join(results_dir, filename))
                deleted_count += 1
    
    return jsonify({'success': True, 'deleted': deleted_count})

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü"""
    return jsonify({'status': 'healthy', 'service': 'backend'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

