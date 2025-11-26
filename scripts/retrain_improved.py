#!/usr/bin/env python3
"""Quick script to retrain model with improved architecture."""
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import TrainConfig, train_model
from src.preprocessing import prepare_dataset, warm_wav2vec_cache

def main():
    data_dir = Path("data")
    model_path = Path("models/faultsense_cnn.pt")
    
    print("=" * 60)
    print("RETRAINING WITH IMPROVED ARCHITECTURE")
    print("=" * 60)
    
    # Backup old model if exists
    if model_path.exists():
        backup_path = Path("models/faultsense_cnn.pt.backup")
        print(f"\n📦 Backing up old model to {backup_path}")
        import shutil
        shutil.copy(model_path, backup_path)
    
    # Prepare dataset
    print("\n📦 Preparing dataset...")
    warm_wav2vec_cache()
    train_split, test_split, _ = prepare_dataset(data_dir)
    
    print(f"\n✅ Dataset prepared:")
    print(f"   - Training samples: {len(train_split.labels)}")
    print(f"   - Test samples: {len(test_split.labels)}")
    
    # Load label map
    label_map_path = data_dir / "artifacts" / "label_to_idx.json"
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text())
    else:
        label_map = {}
    
    # Create improved training configuration
    config = TrainConfig()
    print(f"\n🎯 Improved Training Configuration:")
    print(f"   - Epochs: {config.epochs}")
    print(f"   - Learning Rate: {config.learning_rate}")
    print(f"   - Augmentation: {config.augment_prob}")
    print(f"   - Dropout: {config.dropout}")
    
    # Train
    print(f"\n🏋️ Training model with improved architecture...")
    report = train_model(train_split, test_split, config=config, label_map=label_map)
    
    # Display results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    metrics = report.metrics
    print(f"\n📊 Final Metrics:")
    print(f"   - Validation Accuracy: {metrics.get('val_accuracy', 0):.4f}")
    print(f"   - Validation F1: {metrics.get('val_f1', 0):.4f}")
    print(f"   - Validation Precision: {metrics.get('val_precision', 0):.4f}")
    print(f"   - Validation Recall: {metrics.get('val_recall', 0):.4f}")
    print(f"\n✅ Model saved to: {report.best_model_path}")

if __name__ == "__main__":
    main()

