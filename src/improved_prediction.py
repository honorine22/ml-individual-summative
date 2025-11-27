#!/usr/bin/env python3
"""
Improved prediction service for the enhanced FaultSense model
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simple_prediction import extract_simple_features
from src.preprocessing import FeatureConfig


class ImprovedFaultSenseCNN(nn.Module):
    """
    Improved CNN architecture - must match training architecture exactly
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        # Feature extraction with residual blocks
        self.feature_extractor = nn.Sequential(
            # First block - wide receptive field
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            # Second block with residual connection
            nn.Linear(2048, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            
            # Third block - feature compression
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )
        
        # Classification head with multiple paths
        self.classifier_main = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary classifier for better gradient flow
        self.classifier_aux = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, return_aux=False):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Main classification
        main_output = self.classifier_main(attended_features)
        
        if return_aux and self.training:
            # Auxiliary output for training
            aux_output = self.classifier_aux(features)
            return main_output, aux_output
        
        return main_output


class ImprovedPredictionService:
    """
    Enhanced prediction service with better model architecture and preprocessing
    """
    
    def __init__(self):
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.label_map = None
        self.idx_to_label = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_path: str = "models/faultsense_cnn.pt") -> bool:
        """Load the trained model and associated artifacts."""
        try:
            print(f"🔧 Loading improved model from: {model_path}")
            
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
            model_type = registry.get('model_type', 'FaultSenseCNN')
            
            print(f"📋 Model config: {model_type}, input_dim={input_dim}, classes={num_classes}")
            
            # Create model instance
            if model_type == 'ImprovedFaultSenseCNN':
                self.model = ImprovedFaultSenseCNN(input_dim, num_classes, dropout=0.3)
            else:
                # Fallback to original architecture
                from src.model import FaultSenseCNN
                self.model = FaultSenseCNN(input_dim, num_classes)
            
            # Load model weights
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                print(f"❌ Model file not found: {model_path}")
                return False
                
            state_dict = torch.load(model_path_obj, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler
            artifacts_dir = Path("data/artifacts")
            scaler_mean_path = artifacts_dir / "scaler.mean.npy"
            scaler_scale_path = artifacts_dir / "scaler.scale.npy"
            
            if scaler_mean_path.exists() and scaler_scale_path.exists():
                self.scaler_mean = np.load(scaler_mean_path)
                self.scaler_scale = np.load(scaler_scale_path)
                print(f"✅ Scaler loaded: mean shape {self.scaler_mean.shape}")
            else:
                print("⚠️  Scaler files not found, using default normalization")
                self.scaler_mean = np.zeros(input_dim)
                self.scaler_scale = np.ones(input_dim)
            
            # Load label mapping
            label_path = artifacts_dir / "label_to_idx.json"
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
            
            print("✅ Improved model loaded successfully")
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
        Predict fault type from audio file with improved accuracy.
        
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
            # Extract features using improved method
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
                    "model_type": "improved"
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
        """
        Get top-k predictions with confidence scores.
        
        Args:
            audio_file_path: Path to the audio file
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction dictionaries sorted by confidence
        """
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
        """
        Predict multiple audio files efficiently.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for audio_file in audio_files:
            result = self.predict(audio_file)
            result["file"] = audio_file
            results.append(result)
        
        return results


# Global service instance
_prediction_service = None

def get_prediction_service() -> ImprovedPredictionService:
    """Get the global prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = ImprovedPredictionService()
        _prediction_service.load_model()
    return _prediction_service


if __name__ == "__main__":
    # Test the improved prediction service
    service = ImprovedPredictionService()
    
    if service.load_model():
        print("✅ Model loaded successfully")
        
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
                break
    else:
        print("❌ Failed to load model")
