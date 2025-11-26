#!/usr/bin/env python3
"""
Train a model using only mel spectrogram and MFCC features (no Wav2Vec2).
This ensures training and inference use the same features.
"""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import TrainConfig, train_model, FaultSenseCNN
from src.preprocessing import FeatureConfig, FeatureStore, DatasetSplit
from src.simple_prediction import extract_simple_features
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class SimpleAudioDataset(Dataset):
    """Dataset using simple features (no Wav2Vec2)."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_features_from_curated_data():
    """Extract simple features from curated audio files."""
    print("📂 Loading curated audio data...")
    
    data_dir = Path("data")
    curated_dir = data_dir / "curated"
    
    if not curated_dir.exists():
        raise FileNotFoundError(f"Curated data directory not found: {curated_dir}")
    
    config = FeatureConfig()
    features_list = []
    labels_list = []
    filepaths_list = []
    
    # Define label mapping
    label_map = {
        "electrical_fault": 0,
        "fluid_leak": 1,
        "mechanical_fault": 2,
        "normal_operation": 3
    }
    
    print("🎵 Extracting simple features from audio files...")
    
    for label_name, label_idx in label_map.items():
        label_dir = curated_dir / label_name
        if not label_dir.exists():
            print(f"⚠️  Skipping missing label directory: {label_dir}")
            continue
        
        audio_files = list(label_dir.glob("*.wav"))
        print(f"   - {label_name}: {len(audio_files)} files")
        
        for audio_file in audio_files:
            try:
                # Extract simple features (no Wav2Vec2)
                features = extract_simple_features(audio_file, config)
                features_list.append(features)
                labels_list.append(label_idx)
                filepaths_list.append(str(audio_file))
            except Exception as e:
                print(f"⚠️  Error processing {audio_file}: {e}")
    
    if not features_list:
        raise ValueError("No features extracted! Check curated data directory.")
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    print(f"✅ Extracted features:")
    print(f"   - Total samples: {len(features_array)}")
    print(f"   - Feature dimension: {features_array.shape[1]}")
    print(f"   - Class distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}")
    
    return features_array, labels_array, filepaths_list, label_map


def train_simple_model():
    """Train a model using simple features."""
    print("🎯 Training FaultSense model with simple features (no Wav2Vec2)...")
    
    # Extract features
    features, labels, filepaths, label_map = extract_features_from_curated_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Data split:")
    print(f"   - Training: {len(X_train)} samples")
    print(f"   - Validation: {len(X_val)} samples")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    
    # Save scaler and label map
    artifacts_dir = Path("data/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(artifacts_dir / "scaler.mean.npy", scaler.mean_)
    np.save(artifacts_dir / "scaler.scale.npy", scaler.scale_)
    
    with open(artifacts_dir / "label_to_idx.json", "w") as f:
        json.dump(label_map, f, indent=2)
    
    print(f"💾 Saved scaler and label map to {artifacts_dir}")
    
    # Create datasets
    train_dataset = SimpleAudioDataset(X_train_norm, y_train)
    val_dataset = SimpleAudioDataset(X_val_norm, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_dim = X_train_norm.shape[1]
    num_classes = len(label_map)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = FaultSenseCNN(input_dim, num_classes, dropout=0.3).to(device)
    
    print(f"🤖 Model architecture:")
    print(f"   - Input dimension: {input_dim}")
    print(f"   - Number of classes: {num_classes}")
    print(f"   - Device: {device}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    max_patience = 10
    
    print(f"🏋️ Starting training...")
    
    for epoch in range(100):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_losses.append(loss.item())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_loss = np.mean(val_losses)
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Early stopping and best model saving
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 5 == 0 or patience_counter == 0:
            print(f"Epoch {epoch:3d}: "
                  f"Train Acc={train_acc:.3f} F1={train_f1:.3f} | "
                  f"Val Acc={val_acc:.3f} F1={val_f1:.3f} Loss={val_loss:.3f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch} (best F1: {best_f1:.4f})")
            break
    
    # Save best model
    model.load_state_dict(best_model_state)
    model_path = Path("models/faultsense_cnn.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_state, model_path)
    
    # Save registry
    registry = {
        "best_model": str(model_path),
        "best_f1": best_f1,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "feature_type": "simple_features_no_wav2vec",
        "label_map": label_map
    }
    
    with open("models/registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📊 Best validation F1: {best_f1:.4f}")
    print(f"💾 Model saved to: {model_path}")
    print(f"📝 Registry saved to: models/registry.json")


if __name__ == "__main__":
    train_simple_model()
