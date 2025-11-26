#!/usr/bin/env python3
"""
Train a balanced model without class bias issues.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from src.model import TrainConfig, train_model
from src.preprocessing import prepare_dataset, warm_wav2vec_cache


def main():
    """Train a balanced model with improved configuration."""
    print("🎯 Training balanced FaultSense model...")
    
    # Load data splits
    print("📂 Loading data splits...")
    data_dir = Path("data")
    warm_wav2vec_cache()
    train_split, val_split, _ = prepare_dataset(data_dir)
    
    # Load label map
    label_map_path = data_dir / "artifacts" / "label_to_idx.json"
    label_map = json.loads(label_map_path.read_text()) if label_map_path.exists() else {}
    
    # Improved training configuration - no class bias
    config = TrainConfig(
        epochs=80,
        batch_size=32,
        learning_rate=2e-4,  # Slightly lower learning rate
        use_class_weights=True,  # Still use class weights for balance
        fluid_weight_boost=1.0,  # NO BOOST - this was causing bias!
        augment_prob=0.5,  # Moderate augmentation
        augment_noise_std=0.02,  # Lower noise
        scheduler_patience=8,  # More patience for LR reduction
        early_stopping_patience=15,  # More patience for early stopping
        label_smoothing=0.05,  # Less aggressive label smoothing
        grad_clip=1.0,  # Slightly higher grad clip
        weight_decay=1e-4,  # Lower weight decay to reduce overfitting
        dropout=0.4,  # Lower dropout for better learning
        mlflow_experiment=None,  # Disable MLflow for cleaner training
    )
    
    print(f"🔧 Training configuration:")
    print(f"   - Epochs: {config.epochs}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Fluid weight boost: {config.fluid_weight_boost} (FIXED - was 1.33)")
    print(f"   - Dropout: {config.dropout}")
    print(f"   - Weight decay: {config.weight_decay}")
    print(f"   - Early stopping patience: {config.early_stopping_patience}")
    
    # Train the model
    report = train_model(train_split, val_split, config, label_map)
    
    print(f"\n✅ Training completed!")
    print(f"📊 Final metrics: {report.metrics}")
    print(f"💾 Model saved to: {report.best_model_path}")


if __name__ == "__main__":
    main()
