# 🤖 ML Model Integration - Quick Reference

## ⚡ TL;DR - 3 Steps to Integrate Your Model

### Step 1: Add Model File
```bash
# Copy your model to simulation folder
cp /path/to/your/model.h5 simulation/
cd simulation
```

### Step 2: Edit ml_model.py
Open `simulation/ml_model.py` and update:

**Find:** `def load_model(self):`  
**Replace:** Add your model loading code (see examples below)

**Find:** `def detect(self, image_path):`  
**Replace:** Add your detection code (see examples below)

### Step 3: Run It!
```bash
cd simulation
python detector.py --model your_model.h5
```

**Dashboard updates in real-time at:** http://localhost:8000

---

## 🎯 Code Snippets by Framework

### TensorFlow/Keras

**Load Model:**
```python
def load_model(self):
    import tensorflow as tf
    self.model = tf.keras.models.load_model(self.model_path)
```

**Detect:**
```python
def detect(self, image_path):
    try:
        import cv2
        import numpy as np
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        
        predictions = self.model.predict(np.array([image]))
        confidence = float(np.max(predictions[0]))
        snake_idx = int(np.argmax(predictions[0]))
        snake_id = list(self.snake_classes.keys())[snake_idx]
        
        return {
            'snake_id': snake_id,
            'confidence': confidence,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}
```

---

### PyTorch

**Load Model:**
```python
def load_model(self):
    import torch
    self.model = torch.load(self.model_path)
    self.model.eval()
```

**Detect:**
```python
def detect(self, image_path):
    try:
        import torch
        import cv2
        import torchvision.transforms as transforms
        from PIL import Image
        
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        confidence, snake_idx = torch.max(output, 1)
        snake_id = list(self.snake_classes.keys())[snake_idx.item()]
        
        return {
            'snake_id': snake_id,
            'confidence': float(confidence[0]),
            'success': True
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}
```

---

### YOLO (Ultralytics)

**Load Model:**
```python
def load_model(self):
    from ultralytics import YOLO
    self.model = YOLO(self.model_path)
```

**Detect:**
```python
def detect(self, image_path):
    try:
        results = self.model.predict(image_path, conf=0.5)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            result = results[0]
            box = result.boxes[0]
            snake_name = self.model.names[int(box.cls)]
            confidence = float(box.conf)
            snake_id = self.snake_classes.get(snake_name, 'unknown')
            
            return {
                'snake_id': snake_id,
                'confidence': confidence,
                'success': True
            }
        
        return {'success': False, 'message': 'No detection found'}
    except Exception as e:
        return {'success': False, 'message': str(e)}
```

---

## 🔧 Running Your Model

```bash
# Basic
python detector.py --model your_model.h5

# Custom API
python detector.py --model your_model.h5 --api http://custom:5000

# Confidence threshold
python detector.py --model your_model.h5 --threshold 0.8

# Limit frames
python detector.py --model your_model.h5 --frames 100

# All options
python detector.py --model your_model.h5 --api http://localhost:5000 --threshold 0.75 --frames 200
```

---

## 🐍 Snake Class Names

Map your model's class names to these:

```python
self.snake_classes = {
    'cobra': 'cobra',              # Your model name → Our ID
    'krait': 'viper',              
    'python': 'python',            
    'ratsnake': 'ratsnake',        
    'russell_viper': 'viper_pit',  
    'saw_scaled': 'saw_scaled_viper'
}
```

Edit the **keys** (left side) to match your model's class names.

---

## 🧪 Test Commands

```bash
# Test 1: Load model
python -c "from ml_model import SnakeDetectionModel; m = SnakeDetectionModel('your_model.h5')"

# Test 2: Single image detection
python -c "from ml_model import SnakeDetectionModel; m = SnakeDetectionModel('your_model.h5'); print(m.detect('test.jpg'))"

# Test 3: Live detection (5 frames)
python detector.py --model your_model.h5 --frames 5
```

---

## 📦 Required Packages

```bash
# All frameworks
pip install opencv-python requests

# TensorFlow
pip install tensorflow

# PyTorch
pip install torch torchvision

# YOLO
pip install ultralytics
```

---

## 📊 What Gets Sent to API

```json
{
    "snake_id": "cobra",
    "location": "Camera Feed",
    "confidence": 0.95
}
```

---

## ❓ Common Issues

| Issue | Solution |
|-------|----------|
| "Model file not found" | Copy model to `simulation/` folder |
| "ModuleNotFoundError" | `pip install tensorflow` (or torch, ultralytics) |
| "API connection refused" | Run backend first: `cd backend && python app.py` |
| "No detections" | Lower threshold: `--threshold 0.5` |
| "Wrong snake detected" | Update `snake_classes` mapping |

---

## ✅ Success Checklist

- [ ] Model copied to `simulation/`
- [ ] `ml_model.py` load_model() updated
- [ ] `ml_model.py` detect() updated
- [ ] Backend running (`python app.py`)
- [ ] Test: `python detector.py --model your_model.h5 --frames 5`
- [ ] Dashboard shows detections
- [ ] Confidence scores displayed correctly

---

## 🚀 Full Workflow

```bash
# Terminal 1: Backend
cd backend
python app.py

# Terminal 2: Your ML Model (wait 2 seconds)
cd simulation
python detector.py --model your_model.h5

# Browser: Watch dashboard
http://localhost:8000
```

---

**Ready? Start with Step 1 above! 🐍🤖**
