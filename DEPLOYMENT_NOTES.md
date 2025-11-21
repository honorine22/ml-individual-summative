# Deployment Notes

## Large Files Excluded from Git

The following large files are excluded from the repository via `.gitignore`:

- `data/raw/esc50.zip` (616 MB) - Downloaded automatically during setup
- `mlruns/` (MLflow tracking directory) - Generated during training
- `data/uploads/` - User-uploaded audio files
- Large model checkpoints in `mlruns/`

## Getting Started Without Large Files

1. **Download Data**: Run `python scripts/run_pipeline.py --stage download` to fetch ESC-50 dataset
2. **Train Model**: Run `python scripts/run_pipeline.py --stage train` to generate the model
3. **Model File**: The `models/faultsense_cnn.pt` (44 MB) is included in the repo as it's under GitHub's limit

## For Railway Deployment

The model file and data will be included in the Docker image during build. No additional setup needed.

