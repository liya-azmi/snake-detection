"""
Real-time Snake Detection using Camera + ML Model + API
Replaces the random simulator with actual ML model detection
"""

import requests
import time
import argparse
import json
from pathlib import Path
from ml_model import SnakeDetectionModel

class LiveSnakeDetector:
    """
    Runs ML model on camera feed and sends detections to API
    """
    
    def __init__(self, model_path, api_url="http://localhost:5000"):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained model
            api_url: API server URL
        """
        self.api_url = api_url
        self.model = SnakeDetectionModel(model_path)
        self.detection_count = 0
        
    def is_api_running(self):
        """Check if API server is running"""
        try:
            response = requests.get(f"{self.api_url}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def send_detection_to_api(self, snake_id, location, confidence):
        """Send detection to backend API"""
        try:
            payload = {
                'snake_id': snake_id,
                'location': location,
                'confidence': confidence
            }
            
            response = requests.post(
                f"{self.api_url}/api/detection",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 201:
                result = response.json()
                return True, result
            else:
                return False, f"Status {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def run_camera_detection(self, max_frames=None, confidence_threshold=0.7):
        """
        Run detection on camera stream
        
        Args:
            max_frames: Max frames to process (None = infinite)
            confidence_threshold: Only report detections above this confidence
        """
        
        print("\n" + "="*60)
        print("🐍 LIVE SNAKE DETECTION (ML MODEL)")
        print("="*60)
        print(f"API URL: {self.api_url}")
        print(f"Confidence Threshold: {confidence_threshold * 100}%")
        print("="*60 + "\n")
        
        # Check API
        if not self.is_api_running():
            print("❌ ERROR: API server is not running!")
            print("   Start backend first: cd backend && python app.py")
            return
        
        print("✓ Connected to API server\n")
        
        # Run detection
        frame_count = 0
        try:
            for detection_result in self.model.detect_from_camera(max_detections=max_frames or 1000):
                
                if not detection_result.get('success'):
                    print(f"[Frame {frame_count}] ✗ Detection failed: {detection_result.get('message')}")
                    frame_count += 1
                    continue
                
                snake_id = detection_result['snake_id']
                confidence = detection_result['confidence']
                
                # Only report if above threshold
                if confidence < confidence_threshold:
                    print(f"[Frame {frame_count}] ⊘ {snake_id} ({confidence*100:.1f}%) - Below threshold")
                    frame_count += 1
                    continue
                
                # Send to API
                success, response = self.send_detection_to_api(
                    snake_id=snake_id,
                    location="Camera Feed",
                    confidence=confidence
                )
                
                if success:
                    self.detection_count += 1
                    print(f"[Frame {frame_count}] ✓ Detection #{self.detection_count}")
                    print(f"   Snake: {snake_id}")
                    print(f"   Confidence: {confidence*100:.1f}%")
                    print(f"   API Response: {response.get('message')}")
                else:
                    print(f"[Frame {frame_count}] ✗ Failed to send to API: {response}")
                
                frame_count += 1
                
                # Stop if max frames reached
                if max_frames and frame_count >= max_frames:
                    break
        
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("⏹️  Detection stopped by user")
            print(f"Total detections: {self.detection_count}")
            print("="*60)
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Live Snake Detection using ML Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detector.py --model snake_model.h5
  python detector.py --model model.pth --api http://localhost:5000
  python detector.py --model yolov8.pt --frames 100
  python detector.py --model model.h5 --threshold 0.8
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to trained model file (.h5, .pth, .pt, etc.)'
    )
    parser.add_argument(
        '--api',
        default='http://localhost:5000',
        help='API server URL (default: http://localhost:5000)'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=None,
        help='Max frames to process (default: infinite)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Confidence threshold (0.0-1.0, default: 0.7)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Run detector
    detector = LiveSnakeDetector(model_path=args.model, api_url=args.api)
    detector.run_camera_detection(
        max_frames=args.frames,
        confidence_threshold=args.threshold
    )


if __name__ == '__main__':
    main()
