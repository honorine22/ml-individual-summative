"""Generate EDA visualizations for FaultSense dataset."""
from __future__ import annotations

import random
from pathlib import Path

import librosa
import librosa.display  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_DIR = Path("reports/eda_visuals")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("data/curated/manifest.csv")


if not MANIFEST.exists():
    raise FileNotFoundError("Run the pipeline download stage to create data/curated/manifest.csv")

manifest = pd.read_csv(MANIFEST)

# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------
class_counts = manifest["label"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4))
class_counts.plot(kind="bar", color=["#f94144", "#577590", "#90be6d", "#f3722c"], ax=ax)
ax.set_title("FaultSense class distribution")
ax.set_ylabel("Clip count")
ax.set_xlabel("Fault label")
for idx, value in enumerate(class_counts.values):
    ax.text(idx, value + 1, str(value), ha="center")
fig.tight_layout()
fig.savefig(REPORT_DIR / "class_distribution.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# Waveform + spectrogram for a random clip per class
# ---------------------------------------------------------------------------
random.seed(42)
subset = manifest.groupby("label").sample(1, random_state=42)
fig, axes = plt.subplots(len(subset), 2, figsize=(10, 8))
for row_idx, (_, sample) in enumerate(subset.iterrows()):
    audio_path = Path(sample["filepath"])
    y, sr = librosa.load(audio_path, sr=16000)
    axes[row_idx, 0].plot(y)
    axes[row_idx, 0].set_title(f"Waveform - {sample['label']}")
    axes[row_idx, 0].set_xlim(0, len(y))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=512, ax=axes[row_idx, 1], cmap="magma")
    axes[row_idx, 1].set_title("Log-mel spectrogram")
    fig.colorbar(img, ax=axes[row_idx, 1], format="%+2.0f dB")

fig.tight_layout()
fig.savefig(REPORT_DIR / "sample_waveforms_spectrograms.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# MFCC feature importance proxy (mean magnitude per coefficient per class)
# ---------------------------------------------------------------------------
mfcc_means = []
coeffs = list(range(1, 14))
for label, rows in manifest.groupby("label"):
    mfcc_vectors = []
    for _, row in rows.sample(min(10, len(rows)), random_state=42).iterrows():
        y, sr = librosa.load(row["filepath"], sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=13)
        mfcc_vectors.append(np.mean(np.abs(mfcc), axis=1))
    mfcc_means.append(pd.Series(np.mean(mfcc_vectors, axis=0), index=coeffs, name=label))

mfcc_df = pd.DataFrame(mfcc_means).T
fig, ax = plt.subplots(figsize=(10, 4))
for label in mfcc_df.columns:
    ax.plot(coeffs, mfcc_df[label], label=label)
ax.set_title("Mean MFCC magnitude per class")
ax.set_xlabel("Coefficient index")
ax.set_ylabel("|MFCC| average")
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_DIR / "mfcc_trends.png", dpi=200)
plt.close(fig)

print("EDA visuals written to reports/eda_visuals/")
