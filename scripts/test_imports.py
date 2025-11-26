#!/usr/bin/env python3
"""
Test script to verify all required imports work
Run this after installing requirements-minimal.txt
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports for the API"""
    print("🧪 Testing imports...")
    
    try:
        # Core dependencies
        import numpy as np
        print("✅ numpy")
        
        import pandas as pd
        print("✅ pandas")
        
        import sklearn
        print("✅ scikit-learn")
        
        import torch
        print("✅ torch")
        
        import torchaudio
        print("✅ torchaudio")
        
        import librosa
        print("✅ librosa")
        
        import soundfile as sf
        print("✅ soundfile")
        
        # API dependencies
        import fastapi
        print("✅ fastapi")
        
        import uvicorn
        print("✅ uvicorn")
        
        import pydantic
        print("✅ pydantic")
        
        import requests
        print("✅ requests")
        
        import yaml
        print("✅ pyyaml")
        
        import joblib
        print("✅ joblib")
        
        import tqdm
        print("✅ tqdm")
        
        # Test project imports
        from src.preprocessing import FeatureConfig
        print("✅ src.preprocessing")
        
        from src.model import FaultSenseCNN, TrainConfig
        print("✅ src.model")
        
        from src.prediction import PredictionService
        print("✅ src.prediction")
        
        from src.api import app
        print("✅ src.api")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"\n⚠️  Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
