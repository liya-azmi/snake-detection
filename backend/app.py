from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
from datetime import datetime
import sqlite3
from werkzeug.utils import secure_filename
from snake_classifier import classify_snake_image

app = Flask(__name__)
CORS(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database setup
DB_PATH = '../data/detections.db'

def init_db():
    """Initialize database for storing detection events"""
    if not os.path.exists('../data'):
        os.makedirs('../data')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  snake_id TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  location TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  status TEXT DEFAULT 'detected')''')
    conn.commit()
    conn.close()

# Load snake database
def load_snakes():
    """Load snake database from JSON"""
    try:
        with open('../data/snakes.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# REST API Endpoints

@app.route('/api/snakes', methods=['GET'])
def get_snakes():
    """Get all snakes with details"""
    snakes = load_snakes()
    return jsonify({
        'status': 'success',
        'data': snakes,
        'count': len(snakes)
    })

@app.route('/api/snakes/<snake_id>', methods=['GET'])
def get_snake_details(snake_id):
    """Get specific snake details"""
    snakes = load_snakes()
    if snake_id in snakes:
        return jsonify({
            'status': 'success',
            'data': snakes[snake_id]
        })
    return jsonify({
        'status': 'error',
        'message': f'Snake {snake_id} not found'
    }), 404

@app.route('/api/detection', methods=['POST'])
def report_detection():
    """Report a snake detection from simulation"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['snake_id', 'location', 'confidence']
        if not all(field in data for field in required_fields):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: snake_id, location, confidence'
            }), 400
        
        # Store detection in database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        c.execute('''INSERT INTO detections 
                     (snake_id, timestamp, location, confidence)
                     VALUES (?, ?, ?, ?)''',
                  (data['snake_id'], timestamp, data['location'], data['confidence']))
        
        detection_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': f'Detection recorded',
            'detection_id': detection_id,
            'timestamp': timestamp
        }), 201
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get all recent detections"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('SELECT * FROM detections ORDER BY timestamp DESC LIMIT 50')
        rows = c.fetchall()
        conn.close()
        
        detections = []
        for row in rows:
            detections.append({
                'id': row[0],
                'snake_id': row[1],
                'timestamp': row[2],
                'location': row[3],
                'confidence': row[4],
                'status': row[5]
            })
        
        return jsonify({
            'status': 'success',
            'data': detections,
            'count': len(detections)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/detection/<int:detection_id>/status', methods=['PUT'])
def update_detection_status(detection_id):
    """Update detection status (e.g., 'detected' -> 'notified' -> 'resolved')"""
    try:
        data = request.get_json()
        
        if 'status' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing status field'
            }), 400
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('UPDATE detections SET status = ? WHERE id = ?',
                  (data['status'], detection_id))
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': f'Detection {detection_id} status updated to {data["status"]}'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Snake Detection API'
    })

@app.route('/api/snake_classifier', methods=['POST'])
def snake_classifier():
    """
    Classify uploaded snake images as Venomous or Non-Venomous
    
    Expected multipart/form-data with 'image' file
    
    Returns:
        {
            'status': 'success',
            'classification': 'Venomous' or 'Non Venomous',
            'is_venomous': boolean,
            'confidence': float (0-1),
            'probabilities': {
                'venomous': float,
                'non_venomous': float
            }
        }
    """
    try:
        # Check if image file is in request
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image file provided. Use "image" as the form field name.'
            }), 400
        
        file = request.files['image']
        
        # Check if file was actually selected
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read file content
        file_content = file.read()
        
        # Classify the image
        result = classify_snake_image(file_content)
        
        if not result['success']:
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'Classification failed')
            }), 400
        
        # Store classification result in database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        filename = secure_filename(file.filename)
        
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, f"{datetime.now().timestamp()}_{filename}")
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Insert classification record (using new table)
        c.execute('''CREATE TABLE IF NOT EXISTS classifications
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      filename TEXT NOT NULL,
                      file_path TEXT,
                      classification TEXT NOT NULL,
                      is_venomous BOOLEAN NOT NULL,
                      confidence REAL NOT NULL,
                      venomous_prob REAL,
                      non_venomous_prob REAL,
                      timestamp TEXT NOT NULL)''')
        
        c.execute('''INSERT INTO classifications 
                     (filename, file_path, classification, is_venomous, confidence, 
                      venomous_prob, non_venomous_prob, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (filename, file_path, result['classification'], result['is_venomous'],
                   result['confidence'], result['probabilities']['venomous'],
                   result['probabilities']['non_venomous'], timestamp))
        
        classification_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Return successful classification
        return jsonify({
            'status': 'success',
            'classification_id': classification_id,
            'filename': filename,
            'classification': result['classification'],
            'is_venomous': result['is_venomous'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'timestamp': timestamp
        }), 201
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/classifications', methods=['GET'])
def get_classifications():
    """Get all snake classifications"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''SELECT id, filename, classification, is_venomous, confidence, timestamp 
                     FROM classifications ORDER BY timestamp DESC LIMIT 50''')
        rows = c.fetchall()
        conn.close()
        
        classifications = []
        for row in rows:
            classifications.append({
                'id': row[0],
                'filename': row[1],
                'classification': row[2],
                'is_venomous': bool(row[3]),
                'confidence': row[4],
                'timestamp': row[5]
            })
        
        return jsonify({
            'status': 'success',
            'data': classifications,
            'count': len(classifications)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    init_db()
    print("Starting Snake Detection API on http://localhost:5000")
    print("\n[Available endpoints]")
    print("  GET  /api/snakes - Get all snakes")
    print("  GET  /api/snakes/<id> - Get snake details")
    print("  POST /api/detection - Report snake detection")
    print("  GET  /api/detections - Get all detections")
    print("  PUT  /api/detection/<id>/status - Update detection status")
    print("\n[ML CLASSIFICATION ENDPOINTS]")
    print("  POST /api/snake_classifier - Classify uploaded snake image")
    print("  GET  /api/classifications - Get all classifications")
    print("\n  /api/health - Health check")
    app.run(debug=True, host='0.0.0.0', port=5000)
