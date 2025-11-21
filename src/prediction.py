"""Model inference utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.model import FaultSenseCNN
from src.preprocessing import FeatureConfig, FeatureStore, extract_features


class PredictionService:
    def __init__(self, artifacts_dir: Path, model_path: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = FeatureConfig()
        self.label_to_idx: Dict[str, int] = json.loads((artifacts_dir / "label_to_idx.json").read_text())
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.scaler = FeatureStore.load(artifacts_dir / "scaler.mean.npy")

        input_dim = self.config.n_mels * int(self.config.duration * self.config.sample_rate / self.config.hop_length / 2)
        # fallback to reading from scaler
        input_dim = self.scaler.scaler.mean_.shape[0]
        
        # Try to load model - handle architecture mismatches gracefully
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Detect architecture from checkpoint
        use_old_architecture = False
        if "features.0.weight" in checkpoint:
            first_layer_size = checkpoint["features.0.weight"].shape[0]
            # Old architecture: 1280, New architecture: 1024
            if first_layer_size == 1280:
                use_old_architecture = True
        
        # Create model with appropriate architecture
        if use_old_architecture:
            # Create old architecture model (for backward compatibility)
            import torch.nn as nn
            class OldFaultSenseCNN(nn.Module):
                def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.4):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Linear(input_dim, 1280),
                        nn.BatchNorm1d(1280),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(1280, 640),
                        nn.BatchNorm1d(640),
                        nn.ReLU(),
                        nn.Dropout(dropout * 0.75),
                        nn.Linear(640, 320),
                        nn.BatchNorm1d(320),
                        nn.ReLU(),
                        nn.Dropout(dropout * 0.5),
                        nn.Linear(320, 160),
                        nn.BatchNorm1d(160),
                        nn.ReLU(),
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(dropout * 0.3),
                        nn.Linear(160, num_classes)
                    )
                
                def forward(self, x):
                    features = self.features(x)
                    return self.classifier(features)
            
            self.model = OldFaultSenseCNN(input_dim, len(self.label_to_idx), dropout=0.4)
        else:
            # New architecture (1024)
            self.model = FaultSenseCNN(input_dim, len(self.label_to_idx))
        
        # Load state dict - handle mismatches gracefully
        try:
            self.model.load_state_dict(checkpoint, strict=True)
        except RuntimeError as e:
            # If strict loading fails, try loading only matching keys
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
            unmatched = set(checkpoint.keys()) - set(pretrained_dict.keys())
            if unmatched:
                print(f"Warning: Could not load {len(unmatched)} weights due to architecture mismatch. Model may need retraining.")
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_path: Path) -> Dict[str, float]:
        features = extract_features(audio_path, self.config)
        norm = self.scaler.transform(features.reshape(1, -1))
        tensor = torch.tensor(norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {self.idx_to_label[idx]: float(prob) for idx, prob in enumerate(probs)}

    def predict_top(self, audio_path: Path) -> Dict[str, str | float]:
        dist = self.predict(audio_path)
        label = max(dist, key=dist.get)
        return {"label": label, "confidence": dist[label], "distribution": dist}

    def batch_predict(self, audio_paths: List[Path]) -> List[Dict[str, str | float]]:
        return [self.predict_top(path) for path in audio_paths]

