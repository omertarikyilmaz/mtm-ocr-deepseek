"""
DeepSeek OCR API - Flask uygulama
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ocr_service import SimpleOCRProcessor

app = Flask(__name__)
CORS(app)

# Global processor
processor = None
processing_status = {
    'is_processing': False,
    'progress': 0,
    'total': 0,
    'status_message': 'Hazir'
}

def get_processor():
    """OCR processor'ı lazy load"""
    global processor, processing_status
    if processor is None:
        processing_status['status_message'] = 'Model yukleniyor...'
        processor = SimpleOCRProcessor(
            output_dir='/app/output',
            max_concurrency=30
        )
        processing_status['status_message'] = 'Hazir'
    return processor

@app.route('/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü"""
    return jsonify({'status': 'healthy', 'service': 'deepseek'})

@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    """OCR işleme endpoint'i"""
    global processing_status
    
    if processing_status['is_processing']:
        return jsonify({'error': 'Zaten bir işlem devam ediyor'}), 400
    
    data = request.get_json()
    filenames = data.get('filenames', [])
    
    if not filenames:
        return jsonify({'error': 'İşlenecek dosya bulunamadı'}), 400
    
    # Dosya yolları
    image_paths = [os.path.join('/app/uploads', filename) for filename in filenames]
    image_paths = [p for p in image_paths if os.path.exists(p)]
    
    if not image_paths:
        return jsonify({'error': 'Geçerli dosya bulunamadı'}), 400
    
    try:
        processing_status['is_processing'] = True
        processing_status['total'] = len(image_paths)
        processing_status['progress'] = 0
        processing_status['status_message'] = f'OCR isleniyor... (0/{len(image_paths)})'
        
        # Processor'ı al
        proc = get_processor()
        
        # OCR işle
        results = proc.process_images(image_paths)
        
        processing_status['progress'] = len(results)
        processing_status['status_message'] = f'Tamamlandi: {len(results)} dosya'
        
        return jsonify({
            'success': True,
            'processed': len(results),
            'total': len(image_paths)
        })
        
    except Exception as e:
        processing_status['status_message'] = f'Hata: {str(e)}'
        return jsonify({'error': str(e)}), 500
    
    finally:
        processing_status['is_processing'] = False

@app.route('/api/status', methods=['GET'])
def get_status():
    """İşlem durumu"""
    return jsonify(processing_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)

