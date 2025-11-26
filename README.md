# FaultSense: Audio Fault Classification MLOps Pipeline

An intelligent audio fault classification system that processes equipment sound recordings to predict fault types (mechanical, electrical, fluid leak, or normal operation). Built with PyTorch, FastAPI, and Streamlit for complete MLOps lifecycle management.

## 🎯 Project Overview

This system demonstrates a complete machine learning pipeline for audio classification:
- **Data Processing**: ESC-50 dataset remapped to fault categories
- **Feature Engineering**: Log-mel spectrograms, MFCC, and Wav2Vec2 embeddings
- **Model**: Custom CNN with regularization and class balancing
- **API**: FastAPI endpoints for prediction and retraining
- **UI**: Streamlit dashboard with insights and controls
- **MLOps**: Experiment tracking, model versioning, and automated retraining

## 🏗️ Architecture

- **Data Pipeline**: ESC-50 environmental sounds mapped to fault categories
- **Feature Engineering**: Multi-modal features (spectrograms, MFCC, Wav2Vec2)
- **Model**: PyTorch CNN with dropout, batch normalization, and class weighting
- **API Layer**: FastAPI with prediction, upload, and retraining endpoints
- **User Interface**: Streamlit dashboard for monitoring and control
- **MLOps**: MLflow tracking, model registry, and automated workflows

## 📁 Repository Structure
```
summative/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── notebook/
│   └── faultsense.ipynb        # ML lifecycle documentation
├── src/                        # Core application code
│   ├── preprocessing.py        # Data processing & feature extraction
│   ├── model.py               # CNN architecture & training
│   ├── prediction.py          # Inference service
│   └── api.py                 # FastAPI endpoints
├── app/
│   └── streamlit_app.py       # Web UI dashboard
├── data/                      # Dataset and artifacts
│   ├── train_manifest.csv     # Training data index
│   ├── test_manifest.csv      # Test data index
│   └── uploads/               # User-uploaded files
├── models/
│   ├── registry.json          # Model metadata
│   └── faultsense_cnn.pt     # Trained model weights
├── reports/
│   └── eda_visuals/           # Generated visualizations
└── scripts/
    ├── run_pipeline.py        # Training pipeline
    ├── start.sh              # Application launcher
    └── locustfile.py         # Load testing
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for model training)
- Audio files in WAV format for testing

### Quick Start
```bash
# Clone repository
git clone <your-repo-url>
cd summative

# Install dependencies
pip install -r requirements.txt

# Download and prepare dataset
python scripts/run_pipeline.py --stage download

# Train the model (required - not included in repo)
python scripts/run_pipeline.py --stage train

# Start the application
./scripts/start.sh
```

Access the web interface at http://localhost:8501

### Manual Setup
1. **Environment Setup**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   ```bash
   python scripts/run_pipeline.py --stage download
   ```

3. **Model Training**
   ```bash
   python scripts/run_pipeline.py --stage train --epochs 100
   ```

4. **Run Services**
   ```bash
   # Terminal 1: API Server
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   
   # Terminal 2: Streamlit UI
   streamlit run app/streamlit_app.py --server.port 8501
   ```

## 🔬 Machine Learning Pipeline

The complete ML workflow is documented in `notebook/faultsense.ipynb`:

### 1. Data Acquisition
- Downloads ESC-50 environmental sound dataset
- Maps sounds to fault categories (mechanical, electrical, fluid_leak, normal)
- Creates stratified train/test splits with manifest files

### 2. Feature Engineering
- **Log-mel Spectrograms**: Time-frequency representations
- **MFCC Features**: Mel-frequency cepstral coefficients
- **Wav2Vec2 Embeddings**: Pre-trained audio representations
- **Data Augmentation**: Time stretching, pitch shifting, noise injection

### 3. Model Architecture
- Custom CNN with residual connections
- Dropout and batch normalization for regularization
- Class-weighted loss for imbalanced data
- Early stopping and learning rate scheduling

### 4. Training & Evaluation
- MLflow experiment tracking
- 5 evaluation metrics: Accuracy, Loss, F1, Precision, Recall
- Confusion matrix analysis
- Model checkpointing and registry

### 5. Deployment & Monitoring
- FastAPI REST endpoints
- Streamlit dashboard with real-time metrics
- Automated retraining pipeline

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and system status |
| `/status` | GET | Model metadata and performance metrics |
| `/predict` | POST | Single audio file classification |
| `/upload` | POST | Upload labeled audio for retraining |
| `/retrain` | POST | Trigger model retraining process |

### Example Usage
```bash
# Health check
curl http://localhost:8000/health

# Predict audio file
curl -X POST -F "file=@sample.wav" http://localhost:8000/predict

# Upload training data
curl -X POST -F "file=@fault_sample.wav" -F "label=mechanical_fault" \
     http://localhost:8000/upload
```

## 🎛️ Web Dashboard

The Streamlit interface provides comprehensive system control:

### 📊 Insights Tab
- **Dataset Overview**: Sample counts, class distribution, train/test splits
- **Feature Visualizations**: Waveforms, spectrograms, MFCC trends with interpretations
- **Model Performance**: Training curves, confusion matrix, accuracy metrics
- **Upload Statistics**: Files ready for retraining by category

### 🎯 Prediction Tab
- **Audio Upload**: Drag-and-drop interface for WAV files
- **Real-time Analysis**: Waveform and spectrogram visualization
- **Classification Results**: Confidence scores and probability distributions

### 📤 Upload Data Tab
- **Bulk Upload**: Multiple files with label assignment
- **Progress Tracking**: Upload status and file validation

### 🔄 Retraining Tab
- **One-click Retraining**: Trigger model updates with new data
- **Progress Monitoring**: Real-time training status and metrics
- **Transfer Learning**: Uses existing model as starting point

## 🔄 Retraining Process

The system supports automated model retraining:

1. **Data Upload**: Users upload labeled WAV files via the web interface
2. **Data Storage**: Files are saved to `data/uploads/` with metadata tracking
3. **Trigger Retraining**: One-click button initiates background training process
4. **Model Update**: System fine-tunes existing model with new data
5. **Deployment**: Updated model automatically replaces the current version

### Retraining Features
- **Transfer Learning**: Uses existing model weights as starting point
- **Data Validation**: Ensures uploaded files meet quality standards
- **Progress Tracking**: Real-time updates on training progress
- **Rollback Support**: Previous model versions preserved for safety

## 🧪 Load Testing

Performance testing with Locust framework:

```bash
# Run load test
locust -f scripts/locustfile.py --headless -u 20 -r 5 -t 30s \
       --host http://localhost:8000 --csv=reports/locust/results

# Generate performance plots
python scripts/plot_locust_results.py
```

### Performance Metrics
- **Throughput**: ~16 requests/second
- **Latency**: P95 < 35ms
- **Reliability**: 0% failure rate under normal load
- **Concurrency**: Tested up to 20 concurrent users

## 🚀 Deployment

### Local Development
```bash
# Quick start
./scripts/start.sh

# Access at http://localhost:8501
```

### Cloud Deployment
Recommended platforms:
- **Railway**: Auto-deploy from GitHub
- **Render**: Docker container support
- **Heroku**: Procfile-based deployment
- **Google Cloud Run**: Serverless containers

### Docker
```bash
# Build and run
docker build -t faultsense .
docker run -p 8000:8000 -p 8501:8501 faultsense
```

## 📹 Demo

- **Jupyter Notebook**: Complete ML lifecycle in `notebook/faultsense.ipynb`
- **Video Demo**: [YouTube Link - TBD]
- **Load Test Results**: Performance metrics in `reports/locust/`

## 🎯 Features

✅ **Implemented**
- Audio fault classification (4 categories)
- Real-time prediction API
- Web dashboard with insights
- Automated retraining pipeline
- Load testing and monitoring
- MLflow experiment tracking

🔄 **Future Enhancements**
- Real-time audio streaming
- Mobile app integration
- Advanced noise filtering
- Multi-language support

---

**Built for ALU Machine Learning Pipeline Assignment**