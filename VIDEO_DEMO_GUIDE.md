# Video Demo Guide

This guide will help you record a comprehensive video demonstration that meets all rubric requirements.

## Prerequisites

1. **Start the API server**:
   ```bash
   cd /Users/honorineigiraneza/Documents/ALU/summative
   source .venv/bin/activate
   PYTHONPATH=. uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

2. **Start the Streamlit UI** (in a new terminal):
   ```bash
   cd /Users/honorineigiraneza/Documents/ALU/summative
   source .venv/bin/activate
   streamlit run app/streamlit_app.py --server.port 8501
   ```

3. **Open browser**: Navigate to `http://localhost:8501`

## Video Recording Checklist

### ✅ Part 1: Prediction Process (Required - 5-7 minutes)

**Goal**: Demonstrate correct prediction on a known audio file

1. **Introduction** (30 seconds)
   - Show camera on
   - Introduce yourself and the project
   - Explain: "I'll demonstrate the FaultSense audio fault classification system"

2. **Show the UI** (1 minute)
   - Navigate to the "Predict" tab
   - Explain the interface
   - Show the data insights tab briefly (class distribution, training curves)

3. **Upload and Predict** (2-3 minutes)
   - Use a known test file from `data/test/mechanical_fault/1-64398-B-41.wav`
   - Upload the file
   - Show the waveform and spectrogram visualizations
   - Click "Run Prediction"
   - **IMPORTANT**: Show the prediction result
   - Explain: "The model correctly predicted mechanical_fault with 87% confidence"
   - Show the probability distribution chart

4. **Verify Correctness** (1 minute)
   - Explain: "This file is from the test set and is labeled as mechanical_fault"
   - Show that the prediction matches the actual label
   - This demonstrates CORRECT prediction as required by rubric

### ✅ Part 2: Retraining Process (Required - 5-7 minutes)

**Goal**: Demonstrate full retraining workflow with uploaded data

1. **Upload Data for Retraining** (2 minutes)
   - Navigate to "Upload Data" tab
   - Select a label (e.g., "mechanical_fault")
   - Upload 2-3 test files from `data/test/mechanical_fault/`
   - Show success message: "Files stored. Ready for retraining."
   - **Explain**: "These files are saved to the database for retraining"

2. **Show Data Preprocessing** (1 minute)
   - Navigate to "Insights" tab
   - Show updated class distribution (if visible)
   - **Explain**: "When we trigger retraining, these files will be preprocessed - features extracted, normalized, and incorporated into the training dataset"

3. **Trigger Retraining** (2-3 minutes)
   - Navigate to "Retraining" tab
   - **Explain**: "The retraining process will use our existing model as a pre-trained model - this is transfer learning"
   - Click "Start Retraining"
   - Show the status change to "running"
   - **Explain**: "The system is now:
     1. Loading the existing model as pre-trained
     2. Preprocessing the uploaded data
     3. Fine-tuning the model with the new data"
   - Wait for completion (may take 2-5 minutes)
   - Show completion message and metrics

4. **Show Results** (1 minute)
   - Display the final metrics (accuracy, F1, precision, recall)
   - **Explain**: "The model has been fine-tuned using transfer learning"
   - Show that the model file was updated

### ✅ Part 3: Additional Demonstrations (Optional but Recommended)

1. **Show Data Insights** (1-2 minutes)
   - Navigate to "Insights" tab
   - Show class distribution chart
   - Show training curves (accuracy, F1 over epochs)
   - Show best model performance metrics table
   - **Explain**: "These visualizations help understand the dataset and model performance"

2. **Show API Status** (30 seconds)
   - Show the header metrics (API status, retrain job status)
   - Explain uptime monitoring

## Key Points to Emphasize

### For Prediction (Rubric Requirement):
- ✅ "I'm uploading an audio file for prediction"
- ✅ "The model correctly predicted [label] with [confidence]% confidence"
- ✅ "This matches the actual label from the test set"

### For Retraining (Rubric Requirement):
- ✅ "I'm uploading labeled audio files for retraining"
- ✅ "These files are saved to the database"
- ✅ "The retraining process preprocesses the uploaded data"
- ✅ "The system uses our existing model as a pre-trained model - this is transfer learning"
- ✅ "The model is being fine-tuned with the new data"

## Technical Details to Mention

1. **Pre-trained Model**: "The retraining uses transfer learning - it loads our existing trained model and fine-tunes it with the new data"

2. **Data Preprocessing**: "Uploaded files go through the same preprocessing pipeline: feature extraction (log-mel spectrogram and MFCC), normalization, and train/test splitting"

3. **Evaluation Metrics**: "We track accuracy, precision, recall, F1 score, and loss - all shown in the metrics table"

## Tips for Recording

1. **Camera**: Keep camera on throughout (rubric requirement)
2. **Audio**: Speak clearly, explain what you're doing
3. **Screen**: Show the UI clearly, zoom if needed
4. **Pacing**: Don't rush - explain each step
5. **Testing**: Do a practice run first
6. **Backup**: Have test files ready in a folder for easy access

## Test Files to Use

**For Prediction (High Confidence, Verified Correct)**:
- **PRIMARY**: `data/test/mechanical_fault/1-64398-B-41.wav` → 87.2% confidence, correct ✅
- **BACKUP 1**: `data/test/normal_operation/3-180256-A-0.wav` → 95.1% confidence, correct ✅
- **BACKUP 2**: `data/test/electrical_fault/2-70939-A-42.wav` → 99.5% confidence, correct ✅
- **BACKUP 3**: `data/test/normal_operation/1-97392-A-0.wav` → 90.8% confidence, correct ✅

**For Upload/Retraining**:
- Use any files from `data/test/` directories
- Upload 2-3 files per class for demonstration

**Note**: See `DEMO_FILES.md` for complete list of verified high-confidence files.

## Post-Recording Checklist

1. ✅ Video shows camera on
2. ✅ Prediction process demonstrated with correct result
3. ✅ Upload process shown
4. ✅ Retraining process shown (preprocessing + fine-tuning)
5. ✅ Pre-trained model usage mentioned/explained
6. ✅ Video is clear and audible
7. ✅ Upload to YouTube and add link to README.md

## Quick Start Commands

```bash
# Terminal 1: API Server
cd /Users/honorineigiraneza/Documents/ALU/summative
source .venv/bin/activate
PYTHONPATH=. uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit UI
cd /Users/honorineigiraneza/Documents/ALU/summative
source .venv/bin/activate
streamlit run app/streamlit_app.py --server.port 8501
```

Then open: http://localhost:8501

Good luck with your recording! 🎥

