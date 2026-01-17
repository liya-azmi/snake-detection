import requests
import json
import time
import random
from datetime import datetime

class SnakeDetectionSimulator:
    """Simulates snake detection and sends results to the API"""
    
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.snakes = [
            "cobra", "viper", "python", "ratsnake", 
            "viper_pit", "saw_scaled_viper"
        ]
        self.locations = [
            "Garden Area", "Near Water Body", "Grassland Field",
            "Rocky Terrain", "Agricultural Field", "Forest Edge",
            "Building Foundation", "Walking Path", "Barn Area",
            "River Bank", "Dense Bushes", "Open Field"
        ]
        
    def is_api_running(self):
        """Check if the API server is running"""
        try:
            response = requests.get(f"{self.api_url}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def simulate_detection(self):
        """Simulate a snake detection with random parameters"""
        return {
            'snake_id': random.choice(self.snakes),
            'location': random.choice(self.locations),
            'confidence': round(random.uniform(0.7, 0.99), 2)
        }
    
    def report_detection(self, detection):
        """Send detection to the API"""
        try:
            response = requests.post(
                f"{self.api_url}/api/detection",
                json=detection,
                timeout=5
            )
            
            if response.status_code == 201:
                result = response.json()
                return True, result
            else:
                return False, f"API returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, str(e)
    
    def run_simulation(self, duration_seconds=None, detection_interval=5):
        """
        Run the simulation
        
        Args:
            duration_seconds: How long to run (None = infinite)
            detection_interval: Seconds between simulated detections
        """
        print("\n" + "="*60)
        print("🐍 SNAKE DETECTION SIMULATOR")
        print("="*60)
        print(f"API URL: {self.api_url}")
        print(f"Detection Interval: {detection_interval}s")
        print("="*60 + "\n")
        
        # Check API connection
        if not self.is_api_running():
            print("❌ ERROR: API server is not running!")
            print(f"   Make sure to start the API server first:")
            print(f"   cd backend")
            print(f"   python app.py")
            return
        
        print("✓ Connected to API server\n")
        
        start_time = time.time()
        detection_count = 0
        
        try:
            while True:
                # Check if duration exceeded
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Simulate detection
                detection = self.simulate_detection()
                
                # Report to API
                success, response = self.report_detection(detection)
                
                if success:
                    detection_count += 1
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Detection #{detection_count}")
                    print(f"   Snake: {detection['snake_id']}")
                    print(f"   Location: {detection['location']}")
                    print(f"   Confidence: {detection['confidence']*100:.1f}%")
                    print(f"   Response: {response.get('message')}")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed to report detection")
                    print(f"   Error: {response}")
                
                print()
                
                # Wait before next detection
                time.sleep(detection_interval)
        
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("⏹️  Simulation stopped by user")
            print(f"Total detections: {detection_count}")
            print("="*60)
        except Exception as e:
            print(f"\n❌ Error during simulation: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Snake Detection Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulator.py
  python simulator.py --api http://localhost:5000
  python simulator.py --interval 10
  python simulator.py --duration 60
        """
    )
    
    parser.add_argument(
        '--api',
        default='http://localhost:5000',
        help='API server URL (default: http://localhost:5000)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Detection interval in seconds (default: 5)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration to run in seconds (default: infinite)'
    )
    
    args = parser.parse_args()
    
    simulator = SnakeDetectionSimulator(api_url=args.api)
    simulator.run_simulation(
        duration_seconds=args.duration,
        detection_interval=args.interval
    )

if __name__ == '__main__':
    main()
