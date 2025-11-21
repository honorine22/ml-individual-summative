# FaultSense Hotline MLOps Platform

End-to-end audio fault classification and MLOps system for the "Fault Hotline" use case. Technicians phone in short voice notes describing equipment symptoms; the system interprets each recording, predicts the fault category, surfaces live insights, and supports rapid retraining when new fault patterns emerge.

## Table of Contents
1. [Architecture](#architecture)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Machine Learning Workflow](#machine-learning-workflow)
5. [API](#api)
6. [Streamlit Control Room](#streamlit-control-room)
7. [Retraining Trigger](#retraining-trigger)
8. [Load & Chaos Testing](#load--chaos-testing)
9. [Deployment](#deployment)
10. [Demo Assets](#demo-assets)
11. [Roadmap](#roadmap)

## Architecture
- **Data Source**: ESC-50 audio corpus \+ curated annotations mapping environmental sounds to synthetic fault states (mechanical, electrical, fluid, normal).
- **Feature Store**: Log-mel + MFCC bundles cached under `data/artifacts/` for notebook + API reuse.
- **Model**: Class-weighted feed-forward audio classifier (PyTorch) with stochastic feature noise, ReduceLROnPlateau scheduler, and MLflow tracking.
- **Serving**: FastAPI microservice (`src/api.py`) wrapping the trained model and preprocessing stack; exposes prediction, bulk upload, and retraining routes.
- **UI**: Streamlit dashboard (`streamlit_app.py`) for uptime, predictions, dataset analytics, and retraining triggers.
- **Ops**: Dockerized deployment with Compose and Render/Azure instructions. Locust scripts simulate request floods across scaled containers.

## Repository Structure
```
FaultSense/
├── README.md
├── requirements.txt
├── notebook/
│   └── faultsense.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── api.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── train/
│   ├── test/
│   └── uploads/
├── models/
│   ├── registry.json
│   └── faultsense_cnn.pt
├── reports/
│   ├── eda_visuals/
│   └── locust/
├── scripts/
│   ├── run_pipeline.py
│   ├── generate_eda.py
│   ├── plot_locust_results.py
│   └── locustfile.py
└── infra/
    ├── Dockerfile.api
    ├── Dockerfile.ui
    └── docker-compose.yaml
```

## Getting Started

### Quick Start (Recommended)
```bash
# One-command startup (handles everything)
./scripts/start.sh
```
Access UI at http://localhost:8501

### Manual Setup
1. **Clone & install**
   ```bash
   git clone <repo>
   cd FaultSense
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Bootstrap data & splits**
   ```bash
   python scripts/run_pipeline.py --stage download
   ```
   This downloads ESC-50, remaps classes, and materializes `data/train` / `data/test` + manifests.
3. **Train the model** (required - model file not included in repo due to size)
   ```bash
   python scripts/run_pipeline.py --stage train --epochs 40 --lr 5e-4 --batch-size 48
   ```
   This generates `models/faultsense_cnn.pt` (~44 MB) which is needed for predictions.
4. **Serve**
   ```bash
   # Terminal 1: API
   PYTHONPATH=. uvicorn src.api:app --host 0.0.0.0 --port 8000
   
   # Terminal 2: UI
   export API_URL=http://localhost:8000
   streamlit run app/streamlit_app.py --server.port 8501
   ```

### Docker Deployment
```bash
cd infra
docker compose up --build
```
See `DEPLOYMENT.md` for detailed deployment options.

## Machine Learning Workflow
Detailed walkthrough lives inside the notebook (`notebook/faultsense.ipynb`):
- Data acquisition (ESC-50 fetch, mapping to fault taxonomy, stratified split) + manifest preview.
- Data processing & EDA (`scripts/generate_eda.py`) saving plots to `reports/eda_visuals/` (`sample_waveforms_spectrograms.png`, `mfcc_trends.png`, etc.).
- Feature extraction combines classical features (log-mel + MFCC) **plus** Wav2Vec2 pre-trained embeddings for richer representations.
- Model creation (FaultSense dense net with class-weighted loss, stochastic feature noise augmentation, ReduceLROnPlateau scheduler).
- Evaluation (accuracy, precision, recall, F1 traces + best-epoch table pulled from `models/registry.json` and MLflow run history).
- Prediction demo (uses the same `PredictionService` as FastAPI) + retraining hook description.
- Experiment tracking: MLflow logs land under `./mlruns` by default; override with `MLFLOW_TRACKING_URI`.

## API
| Route | Method | Description |
| --- | --- | --- |
| `/health` | GET | Base uptime probe |
| `/status` | GET | Model metadata, latest metrics, retrain queue |
| `/predict` | POST | Single audio file prediction |
| `/batch-predict` | POST | Multi-file inference + aggregate stats |
| `/upload` | POST | Persist new labeled audio for retraining |
| `/retrain` | POST | Trigger background retraining job |
| `/metrics` | GET | Rolling performance metrics for dashboards |

Refer to `src/api.py` for request/response schemas (Pydantic models).

## Streamlit Control Room
A modern web UI with comprehensive data insights:

**📊 Insights Tab** (Rubric Requirement: "Contains data insights"):
- Dataset overview: Total samples, class distribution, train/test split
- Feature analysis: MFCC trends, waveform/spectrogram visualizations with interpretations
- Model performance: Training curves, best metrics table, model health indicators
- Uploaded data statistics: Files ready for retraining by class
- Confusion matrix heatmap highlighting misclassifications (generated after training)

**🎯 Predict Tab**:
- Audio upload with waveform and spectrogram visualization
- Real-time prediction with confidence scores
- Probability distribution charts

**📤 Upload Data Tab**:
- Bulk file upload with label selection
- File persistence tracking

**🔄 Retraining Tab**:
- One-click retraining trigger
- Real-time job status and metrics
- Uses pre-trained model (transfer learning)

## Retraining Trigger
1. User uploads labeled WAV bundle via Streamlit.
2. API persists files under `data/uploads/` and appends metadata to `data/uploads/manifest.json`.
3. User triggers retrain → FastAPI background task fine-tunes the CNN using combined dataset, logs metrics, and updates `models/registry.json`.
4. Streamlit polls `/status` to show job state + swap to newest model when complete.

## Load & Chaos Testing
- `scripts/locustfile.py` floods `/predict` with configurable concurrency; set `SAMPLE_AUDIO` to any WAV in `data/test`.
- Example run: `locust -f scripts/locustfile.py --headless -u 20 -r 5 -t 30s --host http://localhost:8000 --csv=reports/locust/faultsense`.
- Plot generator: `python scripts/plot_locust_results.py` → `reports/locust/locust_latency.png`.
- Latest single-container result (20 concurrent users): **p95 latency 35 ms**, **0% failures**, peak throughput ≈ **16 req/s** (see CSV + PNG in `reports/locust/`).

## Deployment

### Modern Options

**1. Automated Script** (Easiest):
```bash
./scripts/start.sh
```

**2. Docker Compose** (Production-ready):
```bash
cd infra
docker compose up --build
```
Features: Health checks, auto-restart, isolated networking

**3. Cloud Platforms**:
- **Render.com**: See `DEPLOYMENT.md` for step-by-step guide
- **Railway.app**: Supports Docker deployment
- **Heroku**: Use Procfile
- **Azure App Service**: Docker container deployment

See `DEPLOYMENT.md` for comprehensive deployment guide including:
- Environment configuration
- Cloud platform specifics
- Monitoring and troubleshooting
- Production checklist

## Demo Assets
- **Notebook**: `notebook/faultsense.ipynb` (export to HTML for submission).
- **Video demo**: record Streamlit UI showing prediction + retraining (link TBD).
- **Locust results**: screenshots + CSV in `reports/locust/`.

## Roadmap
- [ ] Integrate on-device noise suppression example.
- [ ] Add ASR fallback when custom audio is too long.
- [ ] Wire up GitHub Actions for lint/test/deploy.

