"""
Locust load testing script for FaultSense API
Simulates flood of prediction requests to test model performance under load
"""
from pathlib import Path
import random
from locust import HttpUser, task, between
import io


class FaultSenseUser(HttpUser):
    """Simulates a user making prediction requests to FaultSense API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts - check if API is healthy"""
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"API health check failed: {response.status_code}")
    
    @task(8)  # 80% of requests will be predictions
    def predict_audio(self):
        """Send prediction request with sample audio file"""
        # Use a small test audio file for load testing
        # In real scenario, you'd have actual audio samples
        fake_audio_data = b"RIFF" + b"0" * 1000  # Minimal WAV-like data
        
        files = {
            "file": ("test_audio.wav", io.BytesIO(fake_audio_data), "audio/wav")
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if "label" in result and "confidence" in result:
                    response.success()
                else:
                    response.failure("Invalid prediction response format")
            else:
                response.failure(f"Prediction failed with status {response.status_code}")
    
    @task(1)  # 10% of requests will be status checks
    def check_status(self):
        """Check API status and model metrics"""
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")
    
    @task(1)  # 10% of requests will be health checks
    def health_check(self):
        """Perform health check"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class HeavyLoadUser(HttpUser):
    """Simulates heavy load user for stress testing"""
    
    wait_time = between(0.1, 0.5)  # Very frequent requests
    weight = 1  # Lower weight = fewer of these users
    
    @task
    def rapid_predictions(self):
        """Rapid-fire prediction requests"""
        fake_audio_data = b"RIFF" + b"0" * 500
        files = {
            "file": ("stress_test.wav", io.BytesIO(fake_audio_data), "audio/wav")
        }
        
        self.client.post("/predict", files=files)


# Custom load test scenarios
class BurstTrafficUser(HttpUser):
    """Simulates burst traffic patterns"""
    
    wait_time = between(0.1, 10)  # Irregular timing
    weight = 1
    
    @task
    def burst_requests(self):
        """Send multiple requests in quick succession"""
        fake_audio_data = b"RIFF" + b"0" * 800
        files = {
            "file": ("burst_test.wav", io.BytesIO(fake_audio_data), "audio/wav")
        }
        
        # Send 3-5 requests rapidly
        for _ in range(random.randint(3, 5)):
            self.client.post("/predict", files=files)
