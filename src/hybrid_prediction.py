"""
Hybrid prediction service that balances speed and accuracy.
Uses cached Wav2Vec2 features for better performance.
"""
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import librosa
import soundfile as sf
from functools import lru_cache

from src.model import FaultSenseCNN
from src.preprocessing import FeatureConfig, FeatureStore


class HybridPredictionService:
    """
    Hybrid prediction service that:
    1. Uses cached Wav2Vec2 embeddings for common audio patterns
    2. Falls back to fast prediction (without Wav2Vec2) for unknown patterns
    3. Provides good balance between speed and accuracy
    """
    
    def __init__(self, artifacts_dir: Path, model_path: Path, device: str | None = None):
        print("🚀 Initializing HybridPredictionService...")
        start_time = time.time()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = FeatureConfig()
        
        # Load label mappings
        self.label_to_idx: Dict[str, int] = json.loads((artifacts_dir / "label_to_idx.json").read_text())
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Load scaler
        self.scaler = FeatureStore.load(artifacts_dir / "scaler.mean.npy")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine input dimension dynamically
        dummy_features = self._extract_features_fast(self._create_dummy_audio())
        input_dim = dummy_features.shape[0]
        
        self.model = FaultSenseCNN(input_dim, len(self.label_to_idx))
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Wav2Vec2 cache
        self._init_wav2vec_cache()
        
        elapsed = time.time() - start_time
        print(f"✅ HybridPredictionService ready in {elapsed:.2f}s")
    
    def _create_dummy_audio(self) -> Path:
        """Create a temporary dummy audio file for dimension calculation."""
        dummy_path = Path("temp_dummy.wav")
        dummy_audio = np.random.randn(self.config.target_samples) * 0.01
        sf.write(dummy_path, dummy_audio, self.config.sample_rate)
        return dummy_path
    
    def _init_wav2vec_cache(self):
        """Initialize Wav2Vec2 cache with common patterns."""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec_model.eval()
            self.use_wav2vec = True
            print("✅ Wav2Vec2 cache initialized")
        except Exception as e:
            print(f"⚠️ Wav2Vec2 not available, using fast mode: {e}")
            self.use_wav2vec = False
    
    @lru_cache(maxsize=100)
    def _get_wav2vec_embedding_cached(self, audio_hash: str, audio_array: tuple) -> np.ndarray:
        """Get Wav2Vec2 embedding with caching for repeated patterns."""
        if not self.use_wav2vec:
            return np.zeros(768)
        
        try:
            # Convert tuple back to numpy array
            audio = np.array(audio_array)
            inputs = self.wav2vec_processor(audio, sampling_rate=self.config.sample_rate, return_tensors="pt")
            with torch.no_grad():
                outputs = self.wav2vec_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception:
            return np.zeros(768)
    
    def _extract_features_fast(self, audio_path: Path) -> np.ndarray:
        """Extract features without Wav2Vec2 (fast mode)."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        if len(y) > self.config.target_samples:
            y = y[:self.config.target_samples]
        else:
            y = np.pad(y, (0, max(0, self.config.target_samples - len(y))), "constant")

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.config.sample_rate, n_mels=self.config.n_mels, 
            n_fft=self.config.n_fft, hop_length=self.config.hop_length
        )
        log_mel = librosa.power_to_db(mel)

        # MFCC
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=self.config.n_mfcc)

        # Wav2Vec2 placeholder (zeros for fast mode)
        wav2vec_embedding = np.zeros(768)

        # Concatenate features
        return np.concatenate([log_mel.flatten(), mfcc.flatten(), wav2vec_embedding])
    
    def _extract_features_hybrid(self, audio_path: Path) -> np.ndarray:
        """Extract features with hybrid approach (cached Wav2Vec2 when possible)."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        if len(y) > self.config.target_samples:
            y = y[:self.config.target_samples]
        else:
            y = np.pad(y, (0, max(0, self.config.target_samples - len(y))), "constant")

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.config.sample_rate, n_mels=self.config.n_mels, 
            n_fft=self.config.n_fft, hop_length=self.config.hop_length
        )
        log_mel = librosa.power_to_db(mel)

        # MFCC
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=self.config.n_mfcc)

        # Wav2Vec2 with caching
        audio_hash = str(hash(tuple(y[:1000])))  # Hash first 1000 samples for caching
        wav2vec_embedding = self._get_wav2vec_embedding_cached(audio_hash, tuple(y))

        # Concatenate features
        return np.concatenate([log_mel.flatten(), mfcc.flatten(), wav2vec_embedding])
    
    def predict(self, audio_path: Path, use_fast_mode: bool = False) -> Dict[str, float]:
        """
        Predict with hybrid approach.
        
        Args:
            audio_path: Path to audio file
            use_fast_mode: If True, skip Wav2Vec2 for speed
        """
        if use_fast_mode:
            features = self._extract_features_fast(audio_path)
        else:
            features = self._extract_features_hybrid(audio_path)
        
        # Normalize and predict
        norm = self.scaler.transform(features.reshape(1, -1))
        tensor = torch.tensor(norm, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        return {self.idx_to_label[idx]: float(prob) for idx, prob in enumerate(probs)}
    
    def predict_top(self, audio_path: Path, use_fast_mode: bool = False) -> Dict[str, str | float]:
        """Get top prediction with confidence."""
        distribution = self.predict(audio_path, use_fast_mode)
        label = max(distribution, key=distribution.get)
        return {
            "label": label,
            "confidence": distribution[label],
            "distribution": distribution
        }
    
    def predict_adaptive(self, audio_path: Path, confidence_threshold: float = 0.7) -> Dict[str, str | float]:
        """
        Adaptive prediction that tries fast mode first, then full mode if confidence is low.
        """
        # Try fast mode first
        fast_result = self.predict_top(audio_path, use_fast_mode=True)
        
        # If confidence is high enough, return fast result
        if fast_result["confidence"] >= confidence_threshold:
            return fast_result
        
        # Otherwise, use full mode for better accuracy
        return self.predict_top(audio_path, use_fast_mode=False)
    
    def batch_predict(self, audio_paths: List[Path], use_fast_mode: bool = False) -> List[Dict[str, str | float]]:
        """Batch prediction."""
        return [self.predict_top(path, use_fast_mode) for path in audio_paths]
    
    def __del__(self):
        """Cleanup dummy files."""
        dummy_path = Path("temp_dummy.wav")
        if dummy_path.exists():
            dummy_path.unlink()
