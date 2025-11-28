"""Production prediction service for the ProductionFaultCNN model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf

from src.preprocessing import FeatureConfig, FeatureStore


class ProductionFaultCNN(nn.Module):
    """
    Production-ready CNN optimized for the specific prediction errors identified
    Must match the architecture used in production_ready_model.py
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        # Feature extraction layers with residual-like connections
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ),
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7)
            ),
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.8)
            )
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Forward through feature layers
        for layer in self.feature_layers:
            x = layer(x)
        
        # Classification
        return self.classifier(x)


def extract_simple_features(audio_path: Path, config: FeatureConfig) -> np.ndarray:
    """Extract only mel spectrogram and MFCC features (no Wav2Vec2). Optimized for speed."""
    # Load audio with faster settings
    # Use shorter duration for faster processing (max 2 seconds)
    max_duration = 2.0
    y, sr = librosa.load(audio_path, sr=config.sample_rate, duration=max_duration, mono=True)
    
    # Pad or truncate to target length (faster than full duration)
    target_samples = int(config.sample_rate * config.duration)
    if len(y) > target_samples:
        y = y[:target_samples]
    elif len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    
    # Extract mel spectrogram with optimized parameters
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=config.n_mfcc)
    
    # Additional spectral features for better discrimination
    # Keep original flattening to maintain 10080 dimensions
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Combine features (NO Wav2Vec2) - maintain original dimension structure
    feat = np.concatenate([
        log_mel.flatten(),
        mfcc.flatten(),
        spectral_centroid.flatten(),
        spectral_rolloff.flatten(),
        zero_crossing_rate.flatten(),
    ])
    
    # Ensure correct dimension (should be 10080)
    target_dim = 10080
    if len(feat) < target_dim:
        feat = np.pad(feat, (0, target_dim - len(feat)), mode='constant')
    elif len(feat) > target_dim:
        feat = feat[:target_dim]
    
    return feat


class ProductionPredictionService:
    """Production prediction service for the ProductionFaultCNN model."""
    
    def __init__(self, artifacts_dir: Path, model_path: Path):
        self.artifacts_dir = artifacts_dir
        self.model_path = model_path
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.label_map = None
        self.idx_to_label = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model automatically
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the production model and artifacts."""
        try:
            print(f"🔧 Loading production model from: {self.model_path}")
            
            # Load registry for model configuration
            registry_path = Path("models/registry.json")
            if not registry_path.exists():
                print("❌ Registry file not found")
                return False
                
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            # Get model configuration
            input_dim = registry.get('input_dim', 10080)
            num_classes = registry.get('num_classes', 4)
            model_type = registry.get('model_type', 'ProductionFaultCNN')
            
            print(f"📋 Model config: {model_type}, input_dim={input_dim}, classes={num_classes}")
            
            # Create model instance with correct architecture
            if model_type == 'ProductionFaultCNN':
                self.model = ProductionFaultCNN(input_dim, num_classes, dropout=0.4)
            else:
                # Fallback to original architecture
                from src.model import FaultSenseCNN
                self.model = FaultSenseCNN(input_dim, num_classes)
            
            # Load model weights
            if not self.model_path.exists():
                print(f"❌ Model file not found: {self.model_path}")
                return False
                
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler
            scaler_mean_path = self.artifacts_dir / "scaler.mean.npy"
            scaler_scale_path = self.artifacts_dir / "scaler.scale.npy"
            
            if scaler_mean_path.exists() and scaler_scale_path.exists():
                self.scaler_mean = np.load(scaler_mean_path)
                self.scaler_scale = np.load(scaler_scale_path)
                print(f"✅ Scaler loaded: mean shape {self.scaler_mean.shape}")
            else:
                print("⚠️  Scaler files not found, using default normalization")
                self.scaler_mean = np.zeros(input_dim)
                self.scaler_scale = np.ones(input_dim)
            
            # Load label mapping
            label_path = self.artifacts_dir / "label_to_idx.json"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    self.label_map = json.load(f)
                self.idx_to_label = {v: k for k, v in self.label_map.items()}
                print(f"✅ Labels loaded: {list(self.label_map.keys())}")
            else:
                # Default labels
                self.label_map = {
                    "electrical_fault": 0,
                    "fluid_leak": 1,
                    "mechanical_fault": 2,
                    "normal_operation": 3
                }
                self.idx_to_label = {v: k for k, v in self.label_map.items()}
                print("⚠️  Using default label mapping")
            
            print("✅ Production model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using the loaded scaler."""
        if self.scaler_mean is None or self.scaler_scale is None:
            print("⚠️  Scaler not loaded, returning raw features")
            return features
            
        # Ensure feature dimensions match
        if len(features) != len(self.scaler_mean):
            print(f"⚠️  Feature dimension mismatch: {len(features)} vs {len(self.scaler_mean)}")
            # Pad or truncate as needed
            if len(features) < len(self.scaler_mean):
                features = np.pad(features, (0, len(self.scaler_mean) - len(features)))
            else:
                features = features[:len(self.scaler_mean)]
        
        # Apply normalization: (x - mean) / scale
        normalized = (features - self.scaler_mean) / self.scaler_scale
        
        # Handle any NaN or inf values
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized
    
    def predict(self, audio_file_path: str) -> Dict:
        """
        Predict fault type from audio file.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                "error": "Model not loaded",
                "label": "unknown",
                "confidence": 0.0,
                "distribution": {}
            }
        
        try:
            # Extract features
            config = FeatureConfig()
            features = extract_simple_features(Path(audio_file_path), config)
            
            if features is None or len(features) == 0:
                return {
                    "error": "Feature extraction failed",
                    "label": "unknown", 
                    "confidence": 0.0,
                    "distribution": {}
                }
            
            # Normalize features
            features_normalized = self._normalize_features(features)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_normalized).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_idx = predicted_idx.item()
                confidence = confidence.item()
                
                # Get all class probabilities
                all_probs = probabilities[0].cpu().numpy()
                
                # Create distribution
                distribution = {}
                for label, idx in self.label_map.items():
                    distribution[label] = float(all_probs[idx])
                
                # Get predicted label
                predicted_label = self.idx_to_label.get(predicted_idx, "unknown")
                
                return {
                    "label": predicted_label,
                    "confidence": confidence,
                    "distribution": distribution,
                    "model_type": "production"
                }
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "label": "unknown",
                "confidence": 0.0,
                "distribution": {}
            }
    
    def predict_top(self, audio_file_path: str, top_k: int = 3) -> List[Dict]:
        """Get top-k predictions with confidence scores."""
        result = self.predict(audio_file_path)
        
        if "error" in result:
            return [result]
        
        # Sort by confidence
        distribution = result["distribution"]
        sorted_predictions = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        top_predictions = []
        for i, (label, confidence) in enumerate(sorted_predictions[:top_k]):
            top_predictions.append({
                "rank": i + 1,
                "label": label,
                "confidence": confidence
            })
        
        return top_predictions
    
    def batch_predict(self, audio_files: List[str]) -> List[Dict]:
        """Predict multiple audio files efficiently."""
        results = []
        
        for audio_file in audio_files:
            result = self.predict(audio_file)
            result["file"] = audio_file
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test the production prediction service
    artifacts_dir = Path("data/artifacts")
    model_path = Path("models/faultsense_cnn.pt")
    
    service = ProductionPredictionService(artifacts_dir, model_path)
    
    # Test with a sample file if available
    test_files = [
        "data/curated/normal_operation/1-30344-A-0.wav",
        "data/curated/electrical_fault/4-90014-B-42.wav",
        "data/curated/fluid_leak/1-28135-B-11.wav",
        "data/curated/mechanical_fault/1-64398-B-41.wav"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n🧪 Testing: {test_file}")
            result = service.predict(test_file)
            print(f"   Prediction: {result['label']} ({result['confidence']:.3f})")
            print(f"   Distribution: {result['distribution']}")
            break
