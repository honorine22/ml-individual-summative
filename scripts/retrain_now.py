#!/usr/bin/env python3
"""Quick script to retrain the model with uploaded data."""
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import retrain_with_new_data, TrainConfig
from src.preprocessing import prepare_dataset_with_uploads, warm_wav2vec_cache

BASE_DIR = Path("data")

print("="*70)
print("FaultSense Model Retraining")
print("="*70)
print()

# Check for uploaded files
upload_manifest = BASE_DIR / "uploads" / "manifest.json"
if upload_manifest.exists():
    import json
    upload_data = json.loads(upload_manifest.read_text())
    print(f"✅ Found {len(upload_data)} uploaded file(s)")
    for record in upload_data:
        print(f"   - {Path(record['filepath']).name} ({record['label']})")
else:
    print("⚠️  No uploaded files found. Retraining with base dataset only.")
print()

# Configure training
config = TrainConfig(
    epochs=80,
    batch_size=32,
    learning_rate=4e-4,
    augment_prob=0.7,
    early_stopping_patience=8,
    dropout=0.55,
    weight_decay=3e-4,
)

print("Training configuration:")
print(f"   - Epochs: {config.epochs}")
print(f"   - Batch size: {config.batch_size}")
print(f"   - Learning rate: {config.learning_rate}")
print(f"   - Dropout: {config.dropout}")
print(f"   - Weight decay: {config.weight_decay}")
print(f"   - Augmentation prob: {config.augment_prob}")
print()

print("Starting retraining...")
print("-" * 70)

warm_wav2vec_cache()

try:
    report = retrain_with_new_data(
        BASE_DIR,
        prepare_dataset_with_uploads,
        config=config
    )
    
    print()
    print("="*70)
    print("✅ Retraining Complete!")
    print("="*70)
    print()
    print("Final Metrics:")
    metrics = report.metrics
    print(f"   Train Accuracy: {metrics.get('train_accuracy', 0):.4f}")
    print(f"   Val Accuracy:   {metrics.get('val_accuracy', 0):.4f}")
    print(f"   Train-Val Gap:  {metrics.get('train_accuracy', 0) - metrics.get('val_accuracy', 0):.4f}")
    print(f"   Val F1:         {metrics.get('val_f1', 0):.4f}")
    print()
    
    gap = metrics.get('train_accuracy', 0) - metrics.get('val_accuracy', 0)
    if gap < 0.1:
        print("🎉 Excellent! Train-val gap < 10% - great generalization!")
    elif gap < 0.15:
        print("✅ Good! Train-val gap < 15% - acceptable generalization")
    else:
        print("⚠️  Train-val gap > 15% - may need more regularization")
    
except Exception as e:
    print()
    print("="*70)
    print("❌ Retraining Failed")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

