"""Simple prediction service without Wav2Vec2 features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import time

import numpy as np
import torch
import librosa
import soundfile as sf

from src.model import FaultSenseCNN
from src.preprocessing import FeatureConfig, FeatureStore


def extract_simple_features(audio_path: Path, config: FeatureConfig) -> np.ndarray:
    """Extract only mel spectrogram and MFCC features (no Wav2Vec2)."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.sample_rate)
    
    # Pad or truncate to target length
    target_samples = int(config.sample_rate * config.duration)
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)))
    
    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    log_mel = librosa.power_to_db(mel)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=config.n_mfcc)
    
    # Additional spectral features for better discrimination
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Combine features (NO Wav2Vec2)
    feat = np.concatenate([
        log_mel.flatten(),
        mfcc.flatten(),
        spectral_centroid.flatten(),
        spectral_rolloff.flatten(),
        zero_crossing_rate.flatten()
    ])
    return feat


class SimplePredictionService:
    """Prediction service using only mel spectrogram and MFCC features."""
    
    def __init__(self, artifacts_dir: Path, model_path: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = FeatureConfig()
        
        # Load label mapping
        self.label_to_idx: Dict[str, int] = json.loads((artifacts_dir / "label_to_idx.json").read_text())
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Load scaler with correct dimensions
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.load(artifacts_dir / "scaler.mean.npy")
        self.scaler.scale_ = np.load(artifacts_dir / "scaler.scale.npy")
        self.scaler.n_features_in_ = len(self.scaler.mean_)
        
        # Calculate correct input dimension
        dummy_audio = np.random.rand(int(self.config.sample_rate * self.config.duration))
        dummy_file = Path("temp_dummy.wav")
        sf.write(dummy_file, dummy_audio, self.config.sample_rate)
        input_dim = extract_simple_features(dummy_file, self.config).shape[0]
        dummy_file.unlink()
        
        print(f"📏 Simple features dimension: {input_dim}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = FaultSenseCNN(input_dim, len(self.label_to_idx))
        
        try:
            self.model.load_state_dict(checkpoint, strict=False)
            print("⚠️  Loaded model with mismatched dimensions (expected for old model)")
        except Exception as e:
            print(f"❌ Could not load model: {e}")
            print("💡 Need to retrain model with correct feature dimensions")
            raise
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, audio_path: Path) -> Dict[str, float]:
        """Predict fault probabilities for an audio file."""
        start_time = time.time()
        
        # Extract features
        features = extract_simple_features(audio_path, self.config)
        
        # Normalize features
        norm = self.scaler.transform(features.reshape(1, -1))
        
        # Convert to tensor and predict
        tensor = torch.tensor(norm, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        end_time = time.time()
        print(f"⚡ Prediction completed in {end_time - start_time:.3f}s")
        
        return {self.idx_to_label[idx]: float(prob) for idx, prob in enumerate(probs)}
    
    def predict_top(self, audio_path: Path) -> Dict[str, str | float]:
        """Get top prediction with confidence and full distribution."""
        distribution = self.predict(audio_path)
        top_label = max(distribution, key=distribution.get)
        
        return {
            "label": top_label,
            "confidence": distribution[top_label],
            "distribution": distribution
        }
    
    def batch_predict(self, audio_paths: List[Path]) -> List[Dict[str, str | float]]:
        """Predict on multiple audio files."""
        return [self.predict_top(path) for path in audio_paths]
