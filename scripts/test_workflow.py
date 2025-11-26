#!/usr/bin/env python3
"""Test script to verify prediction and retraining workflows."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000"


def test_prediction(test_file: Path, expected_label: str = None):
    """Test prediction endpoint with a known file."""
    print(f"\n{'='*60}")
    print(f"Testing Prediction: {test_file.name}")
    print(f"{'='*60}")
    
    try:
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "audio/wav")}
            resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            
            print(f"✅ Prediction successful!")
            print(f"   Predicted: {result['label']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Distribution:")
            for label, prob in sorted(result['distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"     - {label}: {prob:.2%}")
            
            if expected_label:
                if result['label'] == expected_label:
                    print(f"   ✅ CORRECT prediction (expected {expected_label})")
                    return True
                else:
                    print(f"   ⚠️  INCORRECT prediction (expected {expected_label}, got {result['label']})")
                    return False
            return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_upload(file: Path, label: str):
    """Test upload endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing Upload: {file.name} as {label}")
    print(f"{'='*60}")
    
    try:
        with open(file, "rb") as f:
            files = [("files", (file.name, f, "audio/wav"))]
            resp = requests.post(f"{API_URL}/upload", params={"label": label}, files=files, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            
            print(f"✅ Upload successful!")
            print(f"   Stored files: {len(result['stored_files'])}")
            for path in result['stored_files']:
                print(f"     - {path}")
            return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_retrain():
    """Test retraining endpoint."""
    print(f"\n{'='*60}")
    print("Testing Retraining")
    print(f"{'='*60}")
    
    try:
        # Check status before
        status_before = requests.get(f"{API_URL}/status", timeout=5).json()
        print(f"Status before: {status_before.get('job', {}).get('status', 'unknown')}")
        
        # Trigger retraining
        resp = requests.post(f"{API_URL}/retrain", timeout=5)
        resp.raise_for_status()
        result = resp.json()
        
        print(f"✅ Retraining scheduled!")
        print(f"   Status: {result['status']}")
        
        # Poll for completion
        print("\n   Waiting for retraining to complete...")
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = requests.get(f"{API_URL}/status", timeout=5).json()
            job_status = status.get("job", {}).get("status", "unknown")
            
            if job_status == "idle":
                print(f"   ✅ Retraining completed!")
                metrics = status.get("job", {}).get("metrics", {})
                if metrics:
                    print(f"   Final metrics:")
                    print(f"     - Val Accuracy: {metrics.get('val_accuracy', 0):.4f}")
                    print(f"     - Val F1: {metrics.get('val_f1', 0):.4f}")
                    print(f"     - Val Precision: {metrics.get('val_precision', 0):.4f}")
                    print(f"     - Val Recall: {metrics.get('val_recall', 0):.4f}")
                return True
            elif job_status == "running":
                print(f"   ⏳ Still running... ({int(time.time() - start_time)}s)")
            else:
                print(f"   Status: {job_status}")
            
            time.sleep(5)
        
        print(f"   ⚠️  Timeout waiting for retraining")
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("FaultSense Workflow Test Suite")
    print("="*60)
    
    # Test 1: Predictions with known files
    print("\n📊 TEST 1: Prediction Accuracy")
    test_dir = Path("data/test")
    
    test_cases = [
        (test_dir / "mechanical_fault" / "1-64398-B-41.wav", "mechanical_fault"),
        (test_dir / "electrical_fault" / "1-21935-A-38.wav", "electrical_fault"),
        (test_dir / "fluid_leak" / "1-12653-A-15.wav", "fluid_leak"),
        (test_dir / "normal_operation" / "1-30344-A-0.wav", "normal_operation"),
    ]
    
    correct = 0
    total = 0
    for test_file, expected_label in test_cases:
        if test_file.exists():
            total += 1
            if test_prediction(test_file, expected_label):
                correct += 1
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    print(f"\n📈 Prediction Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # Test 2: Upload functionality
    print("\n📤 TEST 2: Upload Functionality")
    upload_file = test_dir / "mechanical_fault" / "1-64398-B-41.wav"
    if upload_file.exists():
        test_upload(upload_file, "mechanical_fault")
    else:
        print("⚠️  Upload test file not found")
    
    # Test 3: Retraining (optional - takes time)
    if "--full" in sys.argv:
        print("\n🔄 TEST 3: Retraining (Full Test)")
        test_retrain()
    else:
        print("\n🔄 TEST 3: Retraining (Skipped - use --full to run)")
        print("   Note: Retraining takes several minutes")
    
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)


if __name__ == "__main__":
    main()

