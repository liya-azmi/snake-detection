"""
ML Model Integration for Snake Detection
This module loads and runs your trained snake detection model
"""

import os
import numpy as np
from pathlib import Path

class SnakeDetectionModel:
    """
    Wrapper for your trained ML model
    Replace this with your actual model loading code
    """
    
    def __init__(self, model_path):
        """
        Initialize the model
        
        Args:
            model_path: Path to your trained model file
        """
        self.model_path = model_path
        self.model = None
        self.snake_classes = {
            'cobra': 'cobra',
            'krait': 'viper',
            'python': 'python',
            'ratsnake': 'ratsnake',
            'russell_viper': 'viper_pit',
            'saw_scaled': 'saw_scaled_viper'
        }
        
        print(f"Loading model from: {model_path}")
        self.load_model()
    
    def load_model(self):
        """
        Load your trained model
        
        REPLACE THIS WITH YOUR ACTUAL MODEL LOADING CODE
        Examples below:
        """
        
        # ============ OPTION 1: TensorFlow/Keras ============
        # import tensorflow as tf
        # self.model = tf.keras.models.load_model(self.model_path)
        
        # ============ OPTION 2: PyTorch ============
        # import torch
        # self.model = torch.load(self.model_path)
        # self.model.eval()
        
        # ============ OPTION 3: YOLO ============
        # from ultralytics import YOLO
        # self.model = YOLO(self.model_path)
        
        # ============ OPTION 4: OpenCV DNN ============
        # import cv2
        # self.model = cv2.dnn.readNetFromDarknet(
        #     cfg=self.model_path + '.cfg',
        #     weights=self.model_path + '.weights'
        # )
        
        # ============ PLACEHOLDER ============
        print("✓ Model loaded successfully (placeholder)")
        print("  ⚠️ IMPORTANT: Replace load_model() with your actual model loading code")
    
    def detect(self, image_path):
        """
        Run detection on an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: {
                'snake_id': 'cobra',
                'confidence': 0.95,
                'bbox': (x, y, w, h),  # optional
                'success': True
            }
        """
        
        try:
            # ============ REPLACE WITH YOUR DETECTION CODE ============
            
            # Example TensorFlow workflow:
            # import cv2
            # image = cv2.imread(image_path)
            # image = cv2.resize(image, (224, 224))
            # image = image / 255.0
            # predictions = self.model.predict(np.array([image]))
            # snake_idx = np.argmax(predictions[0])
            # confidence = float(predictions[0][snake_idx])
            # snake_id = list(self.snake_classes.keys())[snake_idx]
            
            # Example PyTorch workflow:
            # import torch
            # import torchvision.transforms as transforms
            # from PIL import Image
            # image = Image.open(image_path)
            # transform = transforms.Compose([...])
            # input_tensor = transform(image).unsqueeze(0)
            # with torch.no_grad():
            #     output = self.model(input_tensor)
            # confidence, snake_idx = torch.max(output, 1)
            # snake_id = list(self.snake_classes.keys())[snake_idx.item()]
            
            # Example YOLO workflow:
            # results = self.model.predict(image_path)
            # for result in results:
            #     for detection in result.boxes:
            #         snake_id = self.model.names[int(detection.cls)]
            #         confidence = float(detection.conf)
            
            # ============ PLACEHOLDER RESULT ============
            return {
                'snake_id': 'cobra',  # Replace with your detection
                'confidence': 0.92,    # Replace with your score
                'bbox': None,
                'success': True,
                'message': 'Detection successful'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Detection failed: {str(e)}'
            }
    
    def detect_from_camera(self, camera_id=0, max_detections=10):
        """
        Run detection on camera stream
        
        Args:
            camera_id: Camera device ID (0 for default)
            max_detections: How many frames to process
            
        Yields:
            Detection results
        """
        import cv2
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return
        
        print(f"📷 Camera opened. Processing {max_detections} frames...")
        
        frame_count = 0
        while frame_count < max_detections:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save frame temporarily
            temp_path = f"temp_frame_{frame_count}.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Run detection
            result = self.detect(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            yield result
            frame_count += 1
        
        cap.release()
        print(f"✓ Processed {frame_count} frames")


def get_model_instance(model_path=None):
    """
    Factory function to get model instance
    
    Args:
        model_path: Path to your model file
        
    Returns:
        SnakeDetectionModel instance
    """
    if model_path is None:
        # Look for model in common locations
        possible_paths = [
            'model.h5',
            'model.pth',
            'model.pt',
            'snake_model.h5',
            'snake_detector.h5',
            'yolov8_snake.pt',
            '../models/model.h5',
            '../models/model.pt'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None:
        raise FileNotFoundError("Model file not found. Please provide model_path.")
    
    return SnakeDetectionModel(model_path)


# ============ USAGE EXAMPLES ============

if __name__ == '__main__':
    
    # Example 1: Load model and detect from image
    # model = SnakeDetectionModel('path/to/your/model.h5')
    # result = model.detect('image.jpg')
    # print(result)
    
    # Example 2: Detect from camera
    # model = SnakeDetectionModel('path/to/your/model.h5')
    # for detection in model.detect_from_camera(max_detections=20):
    #     print(f"Detected: {detection['snake_id']} (confidence: {detection['confidence']})")
    
    print("✓ ML Model module ready for integration")
    print("  Edit this file to add your actual model loading code")
