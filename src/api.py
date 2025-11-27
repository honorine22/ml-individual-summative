"""FastAPI service exposing prediction and retraining endpoints."""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import TrainConfig, retrain_with_new_data
from src.preprocessing import FeatureConfig, append_upload_metadata, prepare_dataset_with_uploads
from src.production_prediction import ProductionPredictionService as PredictionService

BASE_DIR = Path("data")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_REGISTRY = Path("models/registry.json")

app = FastAPI(title="FaultSense API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=1)
_prediction_service: PredictionService | None = None
_job_state = {"status": "idle", "message": ""}


class PredictResponse(BaseModel):
    label: str
    confidence: float
    distribution: dict


class UploadResponse(BaseModel):
    stored_files: List[str]
    message: str


class RetrainResponse(BaseModel):
    status: str
    metrics: dict | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prediction_service() -> PredictionService:
    """Lazy load the prediction service on first request to save memory."""
    global _prediction_service
    if _prediction_service is not None:
        return _prediction_service

    if not MODEL_REGISTRY.exists():
        raise RuntimeError("Model registry missing; train the model first.")

    print("📦 Loading prediction model (first request)...")
    registry = json.loads(MODEL_REGISTRY.read_text())
    model_path = Path(registry["best_model"])
    _prediction_service = PredictionService(ARTIFACTS_DIR, model_path)
    print("✅ Model loaded and ready")
    return _prediction_service


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def bootstrap():  # pragma: no cover
    # Lazy loading: Model will be loaded on first prediction request
    # This saves memory during startup (important for Render's 512MB limit)
    # Wav2Vec2 is NOT used in production, so we skip warming that cache
    print("🚀 API starting up (model will load on first request)...")
    if MODEL_REGISTRY.exists():
        print("✅ Model registry found - ready for predictions")
    else:
        print("⚠️  Model registry not found - ensure model is trained")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    registry = json.loads(MODEL_REGISTRY.read_text()) if MODEL_REGISTRY.exists() else {}
    return {"job": _job_state, "model": registry}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    import time
    start_time = time.time()
    
    print(f"📥 Prediction request: {file.filename} ({file.size} bytes)")
    
    service = _load_prediction_service()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    
    try:
        result = service.predict(str(tmp_path))
        total_time = time.time() - start_time
        print(f"✅ Prediction completed in {total_time:.3f}s: {result['label']} ({result['confidence']:.2%})")
        return PredictResponse(**result)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/upload", response_model=UploadResponse)
async def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    # Validate label
    valid_labels = ["mechanical_fault", "electrical_fault", "fluid_leak", "normal_operation"]
    if label not in valid_labels:
        raise HTTPException(status_code=422, detail=f"Invalid label. Must be one of: {valid_labels}")
    
    if not files:
        raise HTTPException(status_code=422, detail="No files provided")
    
    stored = []
    for file in files:
        if not file.filename.lower().endswith('.wav'):
            raise HTTPException(status_code=422, detail=f"File {file.filename} must be a WAV file")
        dest = UPLOAD_DIR / label / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(await file.read())
        stored.append(str(dest))
    append_upload_metadata(
        UPLOAD_DIR / "manifest.json",
        [{"label": label, "filepath": path} for path in stored],
    )
    return UploadResponse(stored_files=stored, message="Files ready for retraining")


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(background: BackgroundTasks):
    if _job_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Retraining already in progress")

    def _job():
        global _prediction_service
        _job_state.update({"status": "running", "message": "Retraining with uploaded data"})
        report = retrain_with_new_data(BASE_DIR, prepare_dataset_with_uploads, TrainConfig())
        _prediction_service = None
        _job_state.update({"status": "idle", "message": "Retraining completed successfully", "metrics": report.metrics})

    background.add_task(_job)
    return RetrainResponse(status="scheduled", metrics=None)


@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    service = _load_prediction_service()
    temp_files = []
    
    # Save all uploaded files first
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            temp_files.append(str(tmp.name))
    
    try:
        # Use the batch prediction method
        responses = service.batch_predict(temp_files)
        return responses
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            Path(temp_file).unlink(missing_ok=True)

