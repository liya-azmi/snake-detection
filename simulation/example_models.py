"""
EXAMPLE: Complete ML Model Integration
Shows how to integrate a TensorFlow snake detection model
Copy this file as reference for your actual implementation
"""

import requests
import time
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# EXAMPLE 1: Simple TensorFlow Model Integration
# ============================================================

class TensorFlowSnakeDetector:
    """Example: TensorFlow/Keras model integration"""
    
    def __init__(self, model_path, api_url="http://localhost:5000"):
        self.model_path = model_path
        self.api_url = api_url
        self.model = tf.keras.models.load_model(model_path)
        
        # Class mapping (customize for your model)
        self.class_to_snake_id = {
            0: 'cobra',
            1: 'viper',
            2: 'python',
            3: 'ratsnake',
            4: 'viper_pit',
            5: 'saw_scaled_viper'
        }
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ API: {api_url}")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess image for model"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = image.astype('float32') / 255.0
        return image
    
    def detect(self, image_path):
        """Run detection on image"""
        try:
            # Preprocess
            image = self.preprocess_image(image_path)
            
            # Predict
            predictions = self.model.predict(np.array([image]), verbose=0)
            
            # Get result
            confidence = float(np.max(predictions[0]))
            class_idx = int(np.argmax(predictions[0]))
            snake_id = self.class_to_snake_id.get(class_idx, 'unknown')
            
            return {
                'snake_id': snake_id,
                'confidence': confidence,
                'class_idx': class_idx,
                'success': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def send_to_api(self, detection_result):
        """Send detection to API"""
        if not detection_result['success']:
            return False, str(detection_result.get('error'))
        
        try:
            payload = {
                'snake_id': detection_result['snake_id'],
                'location': 'Camera Feed',
                'confidence': detection_result['confidence']
            }
            
            response = requests.post(
                f"{self.api_url}/api/detection",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 201:
                return True, response.json()
            return False, f"Status {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def run_camera_detection(self, max_frames=10, confidence_threshold=0.7):
        """Run detection on camera"""
        cap = cv2.VideoCapture(0)
        frame_count = 0
        detection_count = 0
        
        print(f"\n📷 Camera Detection Starting...")
        print(f"Processing {max_frames} frames\n")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame temporarily
            temp_path = f"temp_frame_{frame_count}.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Detect
            result = self.detect(temp_path)
            
            if result['success']:
                confidence = result['confidence']
                snake_id = result['snake_id']
                
                if confidence >= confidence_threshold:
                    success, api_response = self.send_to_api(result)
                    
                    if success:
                        detection_count += 1
                        print(f"[{frame_count}] ✓ Detected: {snake_id} ({confidence*100:.1f}%)")
                    else:
                        print(f"[{frame_count}] ✗ API Error: {api_response}")
                else:
                    print(f"[{frame_count}] ⊘ {snake_id} ({confidence*100:.1f}%) - Below threshold")
            else:
                print(f"[{frame_count}] ✗ Detection failed: {result.get('error')}")
            
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            frame_count += 1
        
        cap.release()
        print(f"\n✓ Completed: {detection_count} detections from {frame_count} frames")


# ============================================================
# EXAMPLE 2: PyTorch Model Integration
# ============================================================

class PyTorchSnakeDetector:
    """Example: PyTorch model integration"""
    
    def __init__(self, model_path, api_url="http://localhost:5000"):
        import torch
        import torchvision.transforms as transforms
        
        self.model_path = model_path
        self.api_url = api_url
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = [
            'cobra', 'krait', 'python', 'ratsnake', 'russell_viper', 'saw_scaled_viper'
        ]
        
        print(f"✓ PyTorch model loaded: {model_path}")
        print(f"✓ Using device: {self.device}")
    
    def detect(self, image_path):
        """Run detection on image"""
        import torch
        from PIL import Image
        
        try:
            image = Image.open(image_path)
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
            
            confidence, class_idx = torch.max(output, 1)
            snake_id = self.class_names[class_idx.item()]
            confidence = float(confidence[0])
            
            return {
                'snake_id': snake_id,
                'confidence': confidence,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ============================================================
# EXAMPLE 3: YOLO Model Integration
# ============================================================

class YOLOSnakeDetector:
    """Example: YOLOv8 model integration"""
    
    def __init__(self, model_path, api_url="http://localhost:5000"):
        from ultralytics import YOLO
        
        self.model_path = model_path
        self.api_url = api_url
        self.model = YOLO(model_path)
        
        print(f"✓ YOLO model loaded: {model_path}")
        print(f"Classes: {self.model.names}")
    
    def detect(self, image_path):
        """Run detection on image"""
        try:
            results = self.model.predict(image_path, conf=0.5, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                if len(result.boxes) > 0:
                    box = result.boxes[0]
                    snake_name = self.model.names[int(box.cls)]
                    confidence = float(box.conf)
                    
                    return {
                        'snake_id': snake_name,
                        'confidence': confidence,
                        'bbox': box.xyxy.tolist()[0],
                        'success': True
                    }
            
            return {'success': False, 'error': 'No detection found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ============================================================
# EXAMPLE 4: Custom Model (Generic)
# ============================================================

class CustomSnakeDetector:
    """Example: Custom model with flexible input/output"""
    
    def __init__(self, model_path, api_url="http://localhost:5000", framework='tensorflow'):
        self.model_path = model_path
        self.api_url = api_url
        self.framework = framework
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load model based on framework"""
        if self.framework == 'tensorflow':
            self.model = tf.keras.models.load_model(self.model_path)
        elif self.framework == 'pytorch':
            import torch
            self.model = torch.load(self.model_path)
            self.model.eval()
        elif self.framework == 'yolo':
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
        print(f"✓ {self.framework.upper()} model loaded")
    
    def detect(self, image_path):
        """Run detection with appropriate framework"""
        if self.framework == 'tensorflow':
            return self._detect_tensorflow(image_path)
        elif self.framework == 'pytorch':
            return self._detect_pytorch(image_path)
        elif self.framework == 'yolo':
            return self._detect_yolo(image_path)
    
    def _detect_tensorflow(self, image_path):
        """TensorFlow detection"""
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        
        predictions = self.model.predict(np.array([image]))
        confidence = float(np.max(predictions[0]))
        class_idx = int(np.argmax(predictions[0]))
        
        return {
            'snake_id': f'snake_{class_idx}',
            'confidence': confidence,
            'success': True
        }
    
    def _detect_pytorch(self, image_path):
        """PyTorch detection"""
        from PIL import Image
        image = Image.open(image_path)
        # Add your preprocessing here
        return {'snake_id': 'unknown', 'confidence': 0.5, 'success': True}
    
    def _detect_yolo(self, image_path):
        """YOLO detection"""
        results = self.model.predict(image_path)
        if results and len(results[0].boxes) > 0:
            return {
                'snake_id': self.model.names[int(results[0].boxes[0].cls)],
                'confidence': float(results[0].boxes[0].conf),
                'success': True
            }
        return {'success': False, 'error': 'No detection'}


# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == '__main__':
    
    # Example 1: TensorFlow
    # detector = TensorFlowSnakeDetector('my_model.h5')
    # detector.run_camera_detection(max_frames=20, confidence_threshold=0.7)
    
    # Example 2: PyTorch
    # detector = PyTorchSnakeDetector('my_model.pth')
    
    # Example 3: YOLO
    # detector = YOLOSnakeDetector('yolov8n.pt')
    
    # Example 4: Generic
    # detector = CustomSnakeDetector('my_model.h5', framework='tensorflow')
    
    print("✓ Example implementations ready")
    print("  See above for TensorFlow, PyTorch, YOLO, or custom implementations")
