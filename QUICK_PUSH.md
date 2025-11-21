# Quick Push Solution

## What Was Fixed

✅ Removed from git tracking:
- `models/faultsense_cnn.pt` (44MB) - Users must train their own
- All `.wav` files in `data/` (170MB) - Generated during setup  
- `mlruns/` directory (503MB) - MLflow tracking
- `data/raw/esc50.zip` (616MB) - Downloaded during setup

✅ Updated `.gitignore` to prevent future issues

## Push Now

The push should work now. If it's still slow, it's because large files are in git history, but they won't be pushed in future commits.

```bash
git push origin main
```

## If Push Still Fails

If you still get errors about large files, the files might be in git history. You can:

1. **Force push** (if you're the only one working on this repo):
   ```bash
   git push origin main --force
   ```

2. **Or use Git LFS** for the model file (if you want to keep it):
   ```bash
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes
   git commit -m "Add Git LFS for model files"
   ```

## For Railway Deployment

The model and data will be generated during the Docker build:
- Model: Train during build or download from a release
- Data: Downloaded via `scripts/run_pipeline.py --stage download`

