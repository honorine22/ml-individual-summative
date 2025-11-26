"""Smart prediction service with heuristic-based classification for demo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import time
import numpy as np
import librosa
import soundfile as sf

from src.model import FaultSenseCNN
from src.preprocessing import FeatureConfig, FeatureStore


def analyze_audio_features(audio_path: Path, config: FeatureConfig) -> Dict[str, float]:
    """Analyze audio features to make intelligent predictions."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.sample_rate)
    
    # Pad or truncate to target length
    target_samples = int(config.sample_rate * config.duration)
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)))
    
    # Extract features for analysis
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.sample_rate, n_mels=config.n_mels,
        n_fft=config.n_fft, hop_length=config.hop_length
    )
    log_mel = librosa.power_to_db(mel)
    
    # Calculate audio characteristics
    rms_energy = np.sqrt(np.mean(y**2))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # High frequency energy (for electrical faults)
    high_freq_energy = np.mean(log_mel[32:, :])  # Upper half of mel bands
    
    # Low frequency energy (for mechanical faults)
    low_freq_energy = np.mean(log_mel[:16, :])   # Lower quarter of mel bands
    
    # Temporal variation (for fluid leaks)
    temporal_variation = np.std(np.mean(log_mel, axis=0))
    
    return {
        'rms_energy': float(rms_energy),
        'spectral_centroid': float(spectral_centroid),
        'zero_crossing_rate': float(zero_crossing_rate),
        'spectral_rolloff': float(spectral_rolloff),
        'high_freq_energy': float(high_freq_energy),
        'low_freq_energy': float(low_freq_energy),
        'temporal_variation': float(temporal_variation)
    }


def heuristic_classification(features: Dict[str, float]) -> Dict[str, float]:
    """Make intelligent predictions based on audio characteristics."""
    
    # Initialize base probabilities
    probs = {
        'mechanical_fault': 0.25,
        'electrical_fault': 0.25,
        'fluid_leak': 0.25,
        'normal_operation': 0.25
    }
    
    # Electrical fault indicators
    if features['high_freq_energy'] > -20:  # High frequency content
        probs['electrical_fault'] += 0.3
        probs['normal_operation'] -= 0.1
    
    if features['zero_crossing_rate'] > 0.15:  # High ZCR suggests electrical noise
        probs['electrical_fault'] += 0.2
        probs['mechanical_fault'] -= 0.1
    
    # Mechanical fault indicators  
    if features['low_freq_energy'] > -15:  # Strong low frequency content
        probs['mechanical_fault'] += 0.3
        probs['normal_operation'] -= 0.1
    
    if features['rms_energy'] > 0.05:  # High energy suggests mechanical activity
        probs['mechanical_fault'] += 0.2
        probs['fluid_leak'] -= 0.1
    
    # Fluid leak indicators
    if features['temporal_variation'] > 2.0:  # Variable temporal patterns
        probs['fluid_leak'] += 0.3
        probs['normal_operation'] -= 0.1
    
    if 1000 < features['spectral_centroid'] < 3000:  # Mid-range frequencies
        probs['fluid_leak'] += 0.2
        probs['electrical_fault'] -= 0.1
    
    # Normal operation indicators
    if features['rms_energy'] < 0.02 and features['temporal_variation'] < 1.0:
        probs['normal_operation'] += 0.4
        probs['mechanical_fault'] -= 0.15
        probs['electrical_fault'] -= 0.15
        probs['fluid_leak'] -= 0.1
    
    # Ensure probabilities are positive and sum to 1
    for key in probs:
        probs[key] = max(0.05, probs[key])  # Minimum 5% probability
    
    total = sum(probs.values())
    for key in probs:
        probs[key] /= total
    
    return probs


class SmartPredictionService:
    """Intelligent prediction service using audio feature analysis."""
    
    def __init__(self, artifacts_dir: Path, model_path: Path, device: str | None = None):
        print("🧠 Initializing SmartPredictionService...")
        start_time = time.time()
        
        self.config = FeatureConfig()
        
        # Load label mappings
        self.label_to_idx: Dict[str, int] = json.loads((artifacts_dir / "label_to_idx.json").read_text())
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        init_time = time.time() - start_time
        print(f"✅ SmartPredictionService ready in {init_time:.2f}s")

    def predict(self, audio_path: Path) -> Dict[str, float]:
        """Smart prediction using audio feature analysis."""
        start_time = time.time()
        
        # Analyze audio features
        features = analyze_audio_features(audio_path, self.config)
        
        # Make intelligent classification
        probs = heuristic_classification(features)
        
        pred_time = time.time() - start_time
        print(f"🧠 Smart prediction completed in {pred_time:.3f}s")
        print(f"   Features: RMS={features['rms_energy']:.4f}, SC={features['spectral_centroid']:.0f}Hz")
        
        return probs

    def predict_top(self, audio_path: Path) -> Dict[str, str | float]:
        """Get top prediction with confidence."""
        dist = self.predict(audio_path)
        label = max(dist, key=dist.get)
        return {"label": label, "confidence": dist[label], "distribution": dist}

    def batch_predict(self, audio_paths: List[Path]) -> List[Dict[str, str | float]]:
        """Batch prediction."""
        return [self.predict_top(path) for path in audio_paths]
