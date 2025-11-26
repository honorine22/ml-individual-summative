"""Fast prediction service optimized for deployment."""
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


def extract_features_fast(audio_path: Path, config: FeatureConfig) -> np.ndarray:
    """Fast feature extraction without Wav2Vec2 for deployment."""
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
    
    # Skip Wav2Vec2 for speed - use zeros as placeholder
    wav2vec_embedding = np.zeros(768)
    
    # Combine features
    feat = np.concatenate([log_mel.flatten(), mfcc.flatten(), wav2vec_embedding])
    return feat


class FastPredictionService:
    """Optimized prediction service for deployment."""
    
    def __init__(self, artifacts_dir: Path, model_path: Path, device: str | None = None):
        print("🚀 Initializing FastPredictionService...")
        start_time = time.time()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = FeatureConfig()
        
        # Load label mappings
        self.label_to_idx: Dict[str, int] = json.loads((artifacts_dir / "label_to_idx.json").read_text())
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Load scaler
        self.scaler = FeatureStore.load(artifacts_dir / "scaler.mean.npy")
        input_dim = self.scaler.scaler.mean_.shape[0]
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle both state dict and full checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and any(k.startswith('features.') for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Create model
        self.model = FaultSenseCNN(input_dim, len(self.label_to_idx))
        
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"⚠️  Loading with partial weights: {e}")
            # Load only matching keys
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        init_time = time.time() - start_time
        print(f"✅ FastPredictionService ready in {init_time:.2f}s")

    def predict(self, audio_path: Path) -> Dict[str, float]:
        """Fast prediction without Wav2Vec2."""
        start_time = time.time()
        
        # Fast feature extraction
        features = extract_features_fast(audio_path, self.config)
        
        # Normalize features
        norm = self.scaler.transform(features.reshape(1, -1))
        tensor = torch.tensor(norm, dtype=torch.float32).to(self.device)
        
        # Model inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        result = {self.idx_to_label[idx]: float(prob) for idx, prob in enumerate(probs)}
        
        pred_time = time.time() - start_time
        print(f"⚡ Prediction completed in {pred_time:.3f}s")
        
        return result

    def predict_top(self, audio_path: Path) -> Dict[str, str | float]:
        """Get top prediction with confidence."""
        dist = self.predict(audio_path)
        label = max(dist, key=dist.get)
        return {"label": label, "confidence": dist[label], "distribution": dist}

    def batch_predict(self, audio_paths: List[Path]) -> List[Dict[str, str | float]]:
        """Batch prediction."""
        return [self.predict_top(path) for path in audio_paths]
