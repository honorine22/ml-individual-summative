"""Data acquisition and preprocessing utilities for FaultSense."""
from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import requests
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

# Optional import for Wav2Vec2 (not used in production)
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ESC50_ZIP = "esc50.zip"

_WAV2VEC_CACHE: Dict[str, torch.nn.Module] = {}


def _get_wav2vec2_model():
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("torchaudio is not available. Wav2Vec2 features are disabled.")
    if "model" not in _WAV2VEC_CACHE:
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model = bundle.get_model()
        model.eval()
        _WAV2VEC_CACHE["model"] = model
        _WAV2VEC_CACHE["bundle"] = bundle
    return _WAV2VEC_CACHE["bundle"], _WAV2VEC_CACHE["model"]


def warm_wav2vec_cache(seconds: float = 0.25) -> None:
    """Download + warm the Wav2Vec2 backbone so first request is fast."""
    try:
        bundle, model = _get_wav2vec2_model()
        sample_rate = getattr(bundle, "sample_rate", 16000)
        num_samples = max(1, int(sample_rate * seconds))
        dummy = torch.randn(1, num_samples)
        with torch.inference_mode():
            model.extract_features(dummy)
        print(f"Wav2Vec2 cache primed ({num_samples} samples @ {sample_rate}Hz).")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: Could not warm Wav2Vec2 cache ({exc}).")

FAULT_MAPPING = {
    # mechanical
    "engine": "mechanical_fault",
    "jackhammer": "mechanical_fault",
    "chainsaw": "mechanical_fault",
    # electrical
    "siren": "electrical_fault",
    "drilling": "electrical_fault",
    "clock_tick": "electrical_fault",
    # fluid
    "water_drops": "fluid_leak",
    "rain": "fluid_leak",
    "sea_waves": "fluid_leak",
    # normal
    "dog": "normal_operation",
    "keyboard_typing": "normal_operation",
    "hand_saw": "normal_operation",
}


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 64
    n_mfcc: int = 13
    hop_length: int = 512
    n_fft: int = 1024
    duration: float = 4.0

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.duration)


@dataclass
class DatasetSplit:
    features: np.ndarray
    labels: np.ndarray
    filepaths: List[str]

    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features, dtype=torch.float32)
        y = torch.tensor(self.labels, dtype=torch.long)
        return x, y


class FaultAudioDataset(Dataset):
    def __init__(
        self,
        split: DatasetSplit,
        augment: bool = False,
        augment_prob: float = 0.6,  # Increased from 0.4
        noise_std: float = 0.02,
        mixup_alpha: float = 0.2,
        time_stretch_rate: float = 0.1,  # New: time stretching
        pitch_shift_semitones: float = 2.0,  # New: pitch shifting
        spec_mask_prob: float = 0.3,  # New: spectral masking
        special_label_idx: Optional[int] = None,
    ):
        self.x, self.y = split.to_tensor()
        self.filepaths = split.filepaths
        self.augment = augment
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.time_stretch_rate = time_stretch_rate
        self.pitch_shift_semitones = pitch_shift_semitones
        self.spec_mask_prob = spec_mask_prob
        self.special_label_idx = special_label_idx
        self.config = FeatureConfig()  # For audio loading if needed

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.y)

    def _apply_spectral_masking(self, xb: torch.Tensor) -> torch.Tensor:
        """Apply frequency and time masking to features (simulating audio augmentation)."""
        if torch.rand(1).item() < self.spec_mask_prob:
            # Frequency masking (mask some features)
            mask_size = int(len(xb) * 0.1)  # Mask 10% of features
            if mask_size > 0:
                mask_start = torch.randint(0, max(1, len(xb) - mask_size), (1,)).item()
                xb[mask_start:mask_start + mask_size] = 0
        return xb

    def _enhance_fluid_signature(self, xb: torch.Tensor) -> torch.Tensor:
        """Boost low-frequency energy + add mild noise to highlight fluid leaks."""
        low_band = max(1, int(len(xb) * 0.15))
        boost = 1.05 + torch.rand(1).item() * 0.1  # up to +15%
        xb[:low_band] = xb[:low_band] * boost
        xb = xb + torch.randn_like(xb) * 0.01
        return xb

    def __getitem__(self, idx: int):  # pragma: no cover
        xb = self.x[idx].clone()
        yb = self.y[idx]
        
        if self.augment and torch.rand(1).item() < self.augment_prob:
            aug_type = torch.rand(1).item()
            
            if aug_type < 0.3 and self.mixup_alpha > 0:
                # Mixup augmentation (30% chance)
                mix_idx = torch.randint(0, len(self.y), (1,)).item()
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                xb = lam * xb + (1 - lam) * self.x[mix_idx]
            elif aug_type < 0.5:
                # Gaussian noise + scaling (20% chance)
                noise = torch.randn_like(xb) * self.noise_std
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.15  # ±7.5% scaling (increased)
                xb = scale * (xb + noise)
            elif aug_type < 0.7:
                # Time stretching simulation via feature scaling (20% chance)
                stretch = 1.0 + (torch.rand(1).item() - 0.5) * self.time_stretch_rate
                # Simulate time stretching by scaling features
                xb = xb * stretch
            elif aug_type < 0.9:
                # Pitch shifting simulation via feature shifting (20% chance)
                shift = (torch.rand(1).item() - 0.5) * self.pitch_shift_semitones * 0.1
                xb = xb + shift
            else:
                # Spectral masking (10% chance)
                xb = self._apply_spectral_masking(xb)

        if (
            self.augment
            and self.special_label_idx is not None
            and int(yb.item()) == int(self.special_label_idx)
        ):
            xb = self._enhance_fluid_signature(xb)
        
        return xb, yb


class FeatureStore:
    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler or StandardScaler()

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(features)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.scaler.mean_)
        np.save(path.with_suffix(".scale.npy"), self.scaler.scale_)

    @classmethod
    def load(cls, path: Path) -> "FeatureStore":
        scaler = StandardScaler()
        scaler.mean_ = np.load(path)
        scaler.scale_ = np.load(path.with_suffix(".scale.npy"))
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
        return cls(scaler=scaler)


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def download_esc50(data_dir: Path) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / ESC50_ZIP

    if not zip_path.exists():
        resp = requests.get(ESC50_URL, timeout=60)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    extract_dir = raw_dir / "ESC-50-master"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
    return extract_dir


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _load_audio(path: Path, config: FeatureConfig) -> np.ndarray:
    y, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    if len(y) < config.target_samples:
        pad = config.target_samples - len(y)
        y = np.pad(y, (0, pad), mode="reflect")
    else:
        y = y[: config.target_samples]
    return y


def extract_features(path: Path, config: FeatureConfig) -> np.ndarray:
    y = _load_audio(path, config)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    log_mel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=config.n_mfcc)
    wav2vec_embedding = []
    try:
        bundle, wav2vec_model = _get_wav2vec2_model()
        waveform = torch.tensor(y).unsqueeze(0)
        with torch.inference_mode():
            features, _ = wav2vec_model.extract_features(waveform)
        # Use the last hidden layer and mean-pool over time
        wav2vec_embedding = features[-1].mean(dim=1).squeeze(0).cpu().numpy()
    except Exception:
        wav2vec_embedding = np.zeros(768)
    feat = np.concatenate([log_mel.flatten(), mfcc.flatten(), wav2vec_embedding])
    return feat


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _build_manifest(esc_dir: Path, dest_dir: Path) -> pd.DataFrame:
    metadata_path = esc_dir / "meta" / "esc50.csv"
    meta = pd.read_csv(metadata_path)
    subset = meta[meta["category"].isin(FAULT_MAPPING.keys())].copy()
    subset["fault_label"] = subset["category"].map(FAULT_MAPPING)

    manifest_records = []
    audio_root = esc_dir / "audio"
    for _, row in subset.iterrows():
        fold_dir = audio_root / f"fold{row['fold']}"
        if fold_dir.exists():
            src = fold_dir / row["filename"]
        else:
            # newer ESC-50 zips flatten the audio folder; fall back to direct lookup
            src = audio_root / row["filename"]
        if not src.exists():
            continue
        label_dir = dest_dir / row["fault_label"]
        label_dir.mkdir(parents=True, exist_ok=True)
        dst = label_dir / row["filename"]
        if not dst.exists():
            shutil.copy(src, dst)
        manifest_records.append(
            {
                "filepath": str(dst),
                "label": row["fault_label"],
                "esc_category": row["category"],
                "fold": row["fold"],
            }
        )
    manifest = pd.DataFrame(manifest_records)
    manifest.to_csv(dest_dir / "manifest.csv", index=False)
    return manifest


def prepare_dataset(
    base_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    config: Optional[FeatureConfig] = None,
) -> Tuple[DatasetSplit, DatasetSplit, FeatureStore]:
    config = config or FeatureConfig()
    raw_dir = download_esc50(base_dir)
    curated_dir = base_dir / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(raw_dir, curated_dir)

    label_to_idx = {label: idx for idx, label in enumerate(sorted(manifest["label"].unique()))}
    manifest["label_id"] = manifest["label"].map(label_to_idx)

    features = []
    for path in manifest["filepath"]:
        features.append(extract_features(Path(path), config))
    feature_matrix = np.vstack(features)

    train_df, test_df = train_test_split(
        manifest,
        test_size=test_size,
        stratify=manifest["label_id"],
        random_state=random_state,
    )

    scaler = FeatureStore()
    train_idx = train_df.index.to_list()
    test_idx = test_df.index.to_list()

    x_train = scaler.fit_transform(feature_matrix[train_idx])
    x_test = scaler.transform(feature_matrix[test_idx])

    train_split = DatasetSplit(x_train, train_df["label_id"].to_numpy(), train_df["filepath"].tolist())
    test_split = DatasetSplit(x_test, test_df["label_id"].to_numpy(), test_df["filepath"].tolist())

    def _materialize(df: pd.DataFrame, target: Path):
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            label_dir = target / row["label"]
            label_dir.mkdir(parents=True, exist_ok=True)
            src_path = Path(row["filepath"])
            dst_path = label_dir / src_path.name
            shutil.copy(src_path, dst_path)

    _materialize(train_df, base_dir / "train")
    _materialize(test_df, base_dir / "test")

    train_df.to_csv(base_dir / "train_manifest.csv", index=False)
    test_df.to_csv(base_dir / "test_manifest.csv", index=False)

    scaler_path = base_dir / "artifacts" / "scaler.mean.npy"
    scaler.save(scaler_path)

    (base_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (base_dir / "artifacts" / "label_to_idx.json").write_text(json.dumps(label_to_idx, indent=2))

    return train_split, test_split, scaler


def save_waveform(wave: np.ndarray, path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, samplerate=sample_rate)


def append_upload_metadata(manifest_path: Path, records: List[Dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
    data.extend(records)
    manifest_path.write_text(json.dumps(data, indent=2))


def prepare_dataset_with_uploads(
    base_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    config: Optional[FeatureConfig] = None,
) -> Tuple[DatasetSplit, DatasetSplit, FeatureStore]:
    """Prepare dataset incorporating uploaded data for retraining."""
    config = config or FeatureConfig()
    
    # Start with base curated dataset
    raw_dir = download_esc50(base_dir)
    curated_dir = base_dir / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(raw_dir, curated_dir)
    
    # Incorporate uploaded data
    upload_manifest_path = base_dir / "uploads" / "manifest.json"
    if upload_manifest_path.exists():
        upload_data = json.loads(upload_manifest_path.read_text())
        upload_records = []
        for record in upload_data:
            filepath = Path(record["filepath"])
            if filepath.exists():
                # Copy to curated directory
                label = record["label"]
                label_dir = curated_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)
                dst = label_dir / filepath.name
                if not dst.exists():
                    shutil.copy(filepath, dst)
                upload_records.append({
                    "filepath": str(dst),
                    "label": label,
                    "esc_category": "uploaded",
                    "fold": 0,
                })
        if upload_records:
            upload_df = pd.DataFrame(upload_records)
            manifest = pd.concat([manifest, upload_df], ignore_index=True)
            print(f"Added {len(upload_records)} uploaded files to dataset")
    
    # Continue with normal processing
    label_to_idx = {label: idx for idx, label in enumerate(sorted(manifest["label"].unique()))}
    manifest["label_id"] = manifest["label"].map(label_to_idx)
    
    features = []
    for path in manifest["filepath"]:
        features.append(extract_features(Path(path), config))
    feature_matrix = np.vstack(features)
    
    train_df, test_df = train_test_split(
        manifest,
        test_size=test_size,
        stratify=manifest["label_id"],
        random_state=random_state,
    )
    
    scaler = FeatureStore()
    train_idx = train_df.index.to_list()
    test_idx = test_df.index.to_list()
    
    x_train = scaler.fit_transform(feature_matrix[train_idx])
    x_test = scaler.transform(feature_matrix[test_idx])
    
    train_split = DatasetSplit(x_train, train_df["label_id"].to_numpy(), train_df["filepath"].tolist())
    test_split = DatasetSplit(x_test, test_df["label_id"].to_numpy(), test_df["filepath"].tolist())
    
    def _materialize(df: pd.DataFrame, target: Path):
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            label_dir = target / row["label"]
            label_dir.mkdir(parents=True, exist_ok=True)
            src_path = Path(row["filepath"])
            dst_path = label_dir / src_path.name
            shutil.copy(src_path, dst_path)
    
    _materialize(train_df, base_dir / "train")
    _materialize(test_df, base_dir / "test")
    
    train_df.to_csv(base_dir / "train_manifest.csv", index=False)
    test_df.to_csv(base_dir / "test_manifest.csv", index=False)
    
    scaler_path = base_dir / "artifacts" / "scaler.mean.npy"
    scaler.save(scaler_path)
    
    (base_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (base_dir / "artifacts" / "label_to_idx.json").write_text(json.dumps(label_to_idx, indent=2))
    
    return train_split, test_split, scaler

