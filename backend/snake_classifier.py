"""
Snake Classification Module
Uses the trained MobileNetV2 model to classify uploaded snake images
as Venomous or Non-Venomous

Note: Requires TensorFlow/Keras. If TensorFlow installation fails on Python 3.14,
use PyTorch or switch to Python 3.11 for best compatibility.
"""

import os
import numpy as np
import io

# Try to import TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("[OK] TensorFlow loaded successfully")
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not available - using mock mode for testing")
    tf = None

from PIL import Image

class SnakeClassifier:
    """
    Handles snake image classification using the trained MobileNetV2 model
    """
    
    def __init__(self, model_path='../snake detection/best_model.keras'):
        """
        Initialize the classifier with the trained model
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.is_ready = False
        self.is_demo_mode = False  # Flag for demo/mock mode
        
        # Image processing parameters (from training notebook)
        self.target_size = (224, 224)  # MobileNetV2 preferred size
        
        # Class mapping from the training data
        # 0 = Non-Venomous, 1 = Venomous (binary classification)
        self.class_names = {
            0: 'Non Venomous',
            1: 'Venomous'
        }
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained keras model"""
        try:
            if not TF_AVAILABLE:
                print("[WARN] TensorFlow not available - using demo mode")
                self.is_ready = True
                self.is_demo_mode = True
                return True
            
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.is_ready = True
                self.is_demo_mode = False
                print(f"[OK] Model loaded successfully from {self.model_path}")
                return True
            else:
                print(f"[WARN] Model file not found at {self.model_path} - using demo mode")
                self.is_ready = True
                self.is_demo_mode = True
                return True
        except Exception as e:
            print(f"[WARN] Error loading model: {str(e)} - using demo mode")
            self.is_ready = True
            self.is_demo_mode = True
            return True
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for model inference
        Handles both file paths and PIL Image objects
        
        Args:
            image_input: Either a file path (str) or PIL Image object
        
        Returns:
            np.ndarray: Preprocessed image ready for model inference
        """
        try:
            # Load image if it's a file path
            if isinstance(image_input, str):
                image = Image.open(image_input)
            else:
                # Assume it's already a PIL Image
                image = image_input
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model's expected input size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype='float32') / 255.0
            
            return image_array
        
        except Exception as e:
            print(f"[ERROR] Error preprocessing image: {str(e)}")
            return None
    
    def classify(self, image_input):
        """
        Classify a snake image as Venomous or Non-Venomous
        
        Args:
            image_input: File path or PIL Image object
        
        Returns:
            dict: Classification results with confidence scores
        """
        if not self.is_ready:
            return {
                'success': False,
                'error': 'Model not loaded or not ready'
            }
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_input)
            
            if processed_image is None:
                return {
                    'success': False,
                    'error': 'Failed to preprocess image'
                }
            
            # If in demo mode (TensorFlow not available), return simulated results
            if self.is_demo_mode:
                import random
                venomous_prob = random.uniform(0.3, 0.95)
                is_venomous = venomous_prob > 0.5
                classification = 'Venomous' if is_venomous else 'Non Venomous'
                confidence = max(venomous_prob, 1.0 - venomous_prob)
                
                return {
                    'success': True,
                    'classification': classification,
                    'is_venomous': is_venomous,
                    'confidence': float(confidence),
                    'probabilities': {
                        'venomous': float(venomous_prob),
                        'non_venomous': float(1.0 - venomous_prob)
                    },
                    'demo_mode': True
                }
            
            # Add batch dimension (model expects batch of images)
            input_batch = np.expand_dims(processed_image, axis=0)
            
            # Run inference
            predictions = self.model.predict(input_batch, verbose=0)
            
            # Extract probability for venomous class (binary classification)
            venomous_probability = float(predictions[0][0])
            non_venomous_probability = 1.0 - venomous_probability
            
            # Determine classification (threshold at 0.5)
            is_venomous = venomous_probability > 0.5
            classification = 'Venomous' if is_venomous else 'Non Venomous'
            
            # Get confidence (higher of the two probabilities)
            confidence = max(venomous_probability, non_venomous_probability)
            
            return {
                'success': True,
                'classification': classification,
                'is_venomous': is_venomous,
                'confidence': float(confidence),
                'probabilities': {
                    'venomous': float(venomous_probability),
                    'non_venomous': float(non_venomous_probability)
                }
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Classification error: {str(e)}'
            }
    
    def classify_from_bytes(self, image_bytes):
        """
        Classify a snake image from raw bytes (e.g., uploaded file)
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            dict: Classification results
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            return self.classify(image)
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing image bytes: {str(e)}'
            }


# Global classifier instance
_classifier = None

def get_classifier():
    """Get or create the global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = SnakeClassifier()
    return _classifier

def classify_snake_image(image_input):
    """
    Convenience function to classify a snake image
    
    Args:
        image_input: File path, PIL Image, or raw bytes
    
    Returns:
        dict: Classification results
    """
    classifier = get_classifier()
    
    if isinstance(image_input, bytes):
        return classifier.classify_from_bytes(image_input)
    else:
        return classifier.classify(image_input)
