# 🤖 Model Directory

## Missing Model File?

The trained model `faultsense_cnn.pt` is **not included in Git** due to its large size (55MB). 

### To get the trained model:

**Option 1: Train from scratch (Recommended)**
```bash
# Train the model locally
PYTHONPATH=. python scripts/run_pipeline.py --stage train
```

**Option 2: Quick training for demo**
```bash
# Faster training with fewer epochs
PYTHONPATH=. python scripts/run_pipeline.py --stage train --epochs 20
```

### What you'll get:
- `faultsense_cnn.pt` - The trained CNN model
- `registry.json` - Model metadata and performance metrics

### Model Performance:
- **Accuracy**: ~75%
- **F1 Score**: ~74%
- **Architecture**: CNN with residual connections
- **Features**: Log-mel spectrograms, MFCC, Wav2Vec2 embeddings

The model will be automatically created when you run the training pipeline! 🚀
