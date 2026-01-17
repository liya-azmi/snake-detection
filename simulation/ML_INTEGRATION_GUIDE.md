# 🤖 ML Model Integration Guide

## 📋 Quick Start

You now have **TWO ways** to run detections:

### Option 1: Random Simulator (Current)
```bash
cd simulation
python simulator.py --interval 5
```

### Option 2: Your ML Model (NEW!)
```bash
cd simulation
python detector.py --model your_model.h5
```

---

## 🔧 Integration Steps

### Step 1: Add Your Model File

Place your trained model in the `simulation/` folder:

```
simulation/
├── simulator.py          (random simulator)
├── detector.py          (ML model detector) ← NEW
├── ml_model.py          (model wrapper) ← NEW
└── your_model.h5        (YOUR MODEL FILE) ← COPY HERE
```

### Step 2: Edit `ml_model.py` - Add Your Model Loading Code

Open [ml_model.py](ml_model.py) and find this section (around line 47):

```python
def load_model(self):
    """Load your trained model"""
    
    # ============ OPTION 1: TensorFlow/Keras ============
    # import tensorflow as tf
    # self.model = tf.keras.models.load_model(self.model_path)
```

**Replace with YOUR model loading code:**

#### For TensorFlow/Keras:
```python
def load_model(self):
    import tensorflow as tf
    self.model = tf.keras.models.load_model(self.model_path)
    print(f"✓ TensorFlow model loaded: {self.model_path}")
```

#### For PyTorch:
```python
def load_model(self):
    import torch
    self.model = torch.load(self.model_path)
    self.model.eval()
    print(f"✓ PyTorch model loaded: {self.model_path}")
```

#### For YOLO:
```python
def load_model(self):
    from ultralytics import YOLO
    self.model = YOLO(self.model_path)
    print(f"✓ YOLO model loaded: {self.model_path}")
```

### Step 3: Edit `ml_model.py` - Add Your Detection Code

Find the `detect()` method (around line 90):

```python
def detect(self, image_path):
    """Run detection on an image"""
    try:
        # REPLACE THIS SECTION WITH YOUR CODE
```

**Add your detection logic:**

#### For TensorFlow/Keras:
```python
def detect(self, image_path):
    try:
        import cv2
        import numpy as np
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        
        # Run prediction
        predictions = self.model.predict(np.array([image]))
        
        # Get result
        snake_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][snake_idx])
        snake_id = list(self.snake_classes.keys())[snake_idx]
        
        return {
            'snake_id': snake_id,
            'confidence': confidence,
            'success': True,
            'message': 'Detection successful'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}
```

#### For YOLO:
```python
def detect(self, image_path):
    try:
        # Run YOLO detection
        results = self.model.predict(image_path, conf=0.5)
        
        if len(results) > 0:
            result = results[0]
            if len(result.boxes) > 0:
                # Get top detection
                box = result.boxes[0]
                snake_name = self.model.names[int(box.cls)]
                confidence = float(box.conf)
                
                # Map to our snake IDs
                snake_id = self.snake_classes.get(snake_name, 'unknown')
                
                return {
                    'snake_id': snake_id,
                    'confidence': confidence,
                    'bbox': box.xyxy.tolist(),
                    'success': True,
                    'message': 'Detection successful'
                }
        
        return {'success': False, 'message': 'No detection found'}
    except Exception as e:
        return {'success': False, 'message': str(e)}
```

---

## 🚀 Running Your Model

### Start Backend (if not already running)
```bash
cd backend
python app.py
```

### Run Your ML Model Detector

**Basic usage:**
```bash
cd simulation
python detector.py --model your_model.h5
```

**With custom API endpoint:**
```bash
python detector.py --model your_model.h5 --api http://your-server:5000
```

**With confidence threshold:**
```bash
python detector.py --model your_model.h5 --threshold 0.8
```

**Limit number of frames:**
```bash
python detector.py --model your_model.h5 --frames 100
```

**All options:**
```bash
python detector.py --model your_model.h5 --api http://localhost:5000 --threshold 0.75 --frames 200
```

---

## 📊 What Gets Sent to API

Your model detection → API gets:

```json
{
    "snake_id": "cobra",
    "location": "Camera Feed",
    "confidence": 0.95
}
```

Then the API:
- Stores in database
- Dashboard updates in real-time
- Notifications trigger
- Audio alerts play

---

## 🎯 Model Class Mapping

Your model should detect these snakes:

```python
self.snake_classes = {
    'cobra': 'cobra',              # Indian Cobra
    'krait': 'viper',              # Common Krait
    'python': 'python',            # Indian Python
    'ratsnake': 'ratsnake',        # Rat Snake
    'russell_viper': 'viper_pit',  # Russell's Viper
    'saw_scaled': 'saw_scaled_viper' # Saw-scaled Viper
}
```

**If your model uses different names**, update this mapping in `ml_model.py`:

```python
self.snake_classes = {
    'your_class_name': 'our_snake_id',
    # ... more mappings
}
```

---

## 🔍 Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"

Install required packages:
```bash
# For TensorFlow
pip install tensorflow opencv-python

# For PyTorch
pip install torch torchvision torchaudio opencv-python

# For YOLO
pip install ultralytics opencv-python

# For all
pip install tensorflow torch torchvision opencv-python ultralytics
```

### "Model file not found"

Make sure model is in the right location:
```bash
cd simulation
ls -la your_model.h5  # Should exist
```

### "API connection refused"

Start backend first:
```bash
cd backend
python app.py
# Wait for it to start (2-3 seconds)
```

### "No detections found"

- Check confidence threshold (lower it with `--threshold 0.5`)
- Verify model is working correctly
- Test with `--frames 10` to process just 10 frames

---

## 🧪 Testing Your Model Integration

### Test 1: Load Model Only
```python
from ml_model import SnakeDetectionModel

model = SnakeDetectionModel('your_model.h5')
print("✓ Model loaded successfully")
```

### Test 2: Test Detection on Single Image
```python
from ml_model import SnakeDetectionModel

model = SnakeDetectionModel('your_model.h5')
result = model.detect('test_image.jpg')
print(result)
# Should print: {'snake_id': 'cobra', 'confidence': 0.92, 'success': True}
```

### Test 3: Full Integration Test
```bash
# Terminal 1
cd backend && python app.py

# Terminal 2 (wait 2 seconds)
cd simulation && python detector.py --model your_model.h5 --frames 5
```

---

## 📈 What You'll See

### Terminal Output:
```
============================================================
🐍 LIVE SNAKE DETECTION (ML MODEL)
============================================================
API URL: http://localhost:5000
Confidence Threshold: 70.0%
============================================================

✓ Connected to API server

[Frame 0] ✓ Detection #1
   Snake: cobra
   Confidence: 95.2%
   API Response: Detection recorded

[Frame 1] ✓ Detection #2
   Snake: python
   Confidence: 87.5%
   API Response: Detection recorded

[Frame 2] ⊘ ratsnake (62.3%) - Below threshold

============================================================
⏹️  Detection stopped by user
Total detections: 2
============================================================
```

### Dashboard Updates:
- Detection count increases
- New alerts appear in real-time
- Snake names shown (from your model)
- Confidence percentages displayed

---

## 🎯 Model Input/Output Requirements

### Input
- Image file path (string)
- Image format: JPG, PNG, etc.
- Size: Any (will be resized by your model)

### Output
- Snake ID: String matching `snake_classes` keys
- Confidence: Float between 0.0 and 1.0
- Return format: Dictionary with keys: `snake_id`, `confidence`, `success`

---

## 🔗 Full Example (TensorFlow)

```python
# ml_model.py - load_model() method

def load_model(self):
    """Load TensorFlow/Keras model"""
    import tensorflow as tf
    
    self.model = tf.keras.models.load_model(self.model_path)
    print(f"✓ Model loaded from: {self.model_path}")

# ml_model.py - detect() method

def detect(self, image_path):
    """Run TensorFlow detection"""
    try:
        import cv2
        import numpy as np
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        
        # Predict
        predictions = self.model.predict(np.array([image]), verbose=0)
        confidence = float(np.max(predictions[0]))
        snake_idx = int(np.argmax(predictions[0]))
        
        # Map index to snake ID
        snake_classes_list = list(self.snake_classes.keys())
        snake_id = self.snake_classes[snake_classes_list[snake_idx]]
        
        return {
            'snake_id': snake_id,
            'confidence': confidence,
            'bbox': None,
            'success': True,
            'message': 'Detection successful'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Detection failed: {str(e)}'
        }
```

---

## ✅ Checklist

- [ ] Model file copied to `simulation/` folder
- [ ] `ml_model.py` updated with your model loading code
- [ ] `ml_model.py` updated with your detection code
- [ ] Snake class mapping updated (if needed)
- [ ] Backend running (`python app.py`)
- [ ] Test: `python detector.py --model your_model.h5 --frames 5`
- [ ] Dashboard shows detections in real-time
- [ ] Alerts and notifications working

---

## 🚀 Next Steps

1. **Integrate your model** using the steps above
2. **Test with a few frames** (`--frames 5`)
3. **Monitor dashboard** for real-time updates
4. **Adjust confidence threshold** as needed
5. **Run continuous detection** (`detector.py` without `--frames`)

---

**Your ML model is ready to detect snakes in real-time! 🐍🤖**

Questions? Check the example code above or review the template in `ml_model.py`
