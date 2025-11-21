"""Model architecture, training, and retraining routines."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader

from src.preprocessing import DatasetSplit, FaultAudioDataset


def _infer_flat_shape(split: DatasetSplit) -> int:
    return split.features.shape[1]


class FaultSenseCNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        # Reduced capacity + increased regularization to prevent overfitting
        self.features = nn.Sequential(
            nn.Linear(input_dim, 1024),  # Reduced from 1280
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),  # Increased from 0.4
            
            nn.Linear(1024, 512),  # Reduced from 640
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.9),  # Increased from 0.75
            
            nn.Linear(512, 256),  # Reduced from 320
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),  # Increased from 0.5
            
            nn.Linear(256, 128),  # Reduced from 160
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Added dropout here too
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.4),  # Increased from 0.3
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # pragma: no cover - simple
        features = self.features(x)
        return self.classifier(features)


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 32  # Slightly smaller batch for better regularization
    learning_rate: float = 4e-4  # Slightly lower LR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: Path = Path("models")
    use_class_weights: bool = True
    augment_prob: float = 0.7  # Heavier augmentation to curb overfitting
    augment_noise_std: float = 0.03
    scheduler_patience: int = 3  # React faster to plateau
    early_stopping_patience: int = 8  # Stop closer to best val metrics
    label_smoothing: float = 0.12
    grad_clip: float = 0.8
    weight_decay: float = 3e-4  # Slightly stronger L2
    dropout: float = 0.55  # More dropout in dense stack
    fluid_label_key: str = "fluid_leak"
    fluid_weight_boost: float = 1.35
    reports_dir: Path = Path("reports")
    mlflow_experiment: Optional[str] = "FaultSense"


@dataclass
class TrainReport:
    metrics: Dict[str, float]
    best_model_path: str
    label_map: Dict[str, int]
    trained_at: float

    def to_json(self) -> str:
        payload = asdict(self)
        payload["trained_at"] = self.trained_at
        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _epoch(dataloader: DataLoader, model: nn.Module, criterion, optimizer=None, device="cpu", grad_clip=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, preds, targets = [], [], []
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        losses.append(loss.item())
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        targets.extend(yb.cpu().numpy())
    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average="macro", zero_division=0),
        "recall": recall_score(targets, preds, average="macro", zero_division=0),
        "f1": f1_score(targets, preds, average="macro", zero_division=0),
    }
    return metrics


def train_model(
    train_split: DatasetSplit,
    val_split: DatasetSplit,
    config: Optional[TrainConfig] = None,
    label_map: Optional[Dict[str, int]] = None,
    pretrained_model: Optional[nn.Module] = None,
) -> TrainReport:
    config = config or TrainConfig()
    label_map = label_map or {}
    fluid_idx = label_map.get(config.fluid_label_key) if label_map else None

    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    input_dim = _infer_flat_shape(train_split)
    num_classes = len(np.unique(train_split.labels))

    if pretrained_model is not None:
        model = pretrained_model
        print("Using pre-trained model for fine-tuning")
    else:
        model = FaultSenseCNN(input_dim, num_classes, dropout=config.dropout).to(config.device)

    if config.use_class_weights:
        class_counts = np.bincount(train_split.labels, minlength=num_classes)
        weights = len(train_split.labels) / (num_classes * class_counts)
        if fluid_idx is not None and config.fluid_weight_boost > 1.0:
            weights[fluid_idx] *= config.fluid_weight_boost
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=config.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.scheduler_patience, factor=0.5
    )

    train_loader = DataLoader(
        FaultAudioDataset(
            train_split,
            augment=True,
            augment_prob=config.augment_prob,
            noise_std=config.augment_noise_std,
            mixup_alpha=0.2,
            special_label_idx=fluid_idx,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch to avoid BatchNorm issues
    )
    val_loader = DataLoader(
        FaultAudioDataset(val_split), 
        batch_size=config.batch_size,
        drop_last=False,  # Keep all validation samples
    )

    best_f1 = -1.0
    best_path = model_dir / "faultsense_cnn.pt"
    history = []
    patience_counter = 0
    best_state = None

    mlflow_run = None
    if config.mlflow_experiment:
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow_run = mlflow.start_run(run_name="faultsense-training")
        mlflow.log_params(
            {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "use_class_weights": config.use_class_weights,
                "augment_prob": config.augment_prob,
                "augment_noise_std": config.augment_noise_std,
                "label_smoothing": config.label_smoothing,
                "early_stopping_patience": config.early_stopping_patience,
                "weight_decay": config.weight_decay,
                "dropout": config.dropout,
            }
        )

    try:
        for epoch in range(1, config.epochs + 1):
            train_metrics = _epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                device=config.device,
                grad_clip=config.grad_clip,
            )
            val_metrics = _epoch(val_loader, model, criterion, device=config.device)
            scheduler.step(val_metrics["f1"])
            record = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            if mlflow_run:
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), best_path)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (best F1: {best_f1:.4f})")
                    break
    finally:
        if mlflow_run:
            mlflow.log_artifact(str(best_path))
            mlflow.end_run()

    # Evaluate best model for confusion matrix
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(config.device)
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    eval_loader = DataLoader(FaultAudioDataset(val_split), batch_size=config.batch_size, drop_last=False)
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb, yb = xb.to(config.device), yb.to(config.device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(yb.cpu().tolist())

    try:
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
        report_dir = Path(config.reports_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        idx_to_label = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
        (report_dir / "confusion_matrix.json").write_text(
            json.dumps(
                {
                    "labels": idx_to_label,
                    "matrix": cm.tolist(),
                },
                indent=2,
            )
        )
    except Exception as exc:
        print(f"Warning: Could not save confusion matrix ({exc})")

    registry = {
        "best_model": str(best_path),
        "history": history,
        "label_map": label_map,
    }
    (model_dir / "registry.json").write_text(json.dumps(registry, indent=2))

    return TrainReport(
        metrics=history[-1],
        best_model_path=str(best_path),
        label_map=label_map,
        trained_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Retraining helper
# ---------------------------------------------------------------------------

def retrain_with_new_data(
    base_dir: Path,
    dataset_builder,
    config: Optional[TrainConfig] = None,
    pretrained_model_path: Optional[Path] = None,
) -> TrainReport:
    """Retrain model using existing model as pre-trained (transfer learning)."""
    train_split, test_split, scaler = dataset_builder(base_dir)
    label_map = json.loads((base_dir / "artifacts" / "label_to_idx.json").read_text())
    
    # Use improved config for better regularization
    config = config or TrainConfig()
    model_dir = Path(config.model_dir)
    input_dim = _infer_flat_shape(train_split)
    num_classes = len(np.unique(train_split.labels))
    
    # Check if we can use pre-trained model
    pretrained_path = pretrained_model_path or (model_dir / "faultsense_cnn.pt")
    use_pretrained = False
    model = None
    
    if pretrained_path.exists():
        try:
            pretrained_state = torch.load(pretrained_path, map_location=config.device)
            # Check architecture compatibility
            if "features.0.weight" in pretrained_state:
                first_layer_size = pretrained_state["features.0.weight"].shape[0]
                # New architecture uses 1024, old uses 1280
                if first_layer_size == 1024:
                    # Architecture matches - can use as pre-trained
                    model = FaultSenseCNN(input_dim, num_classes, dropout=config.dropout).to(config.device)
                    model.load_state_dict(pretrained_state, strict=True)
                    use_pretrained = True
                    print(f"✅ Loaded pre-trained model from {pretrained_path} (architecture matches)")
                else:
                    print(f"⚠️  Architecture mismatch detected (old: {first_layer_size}, new: 1024).")
                    print(f"   Training from scratch with improved architecture to reduce overfitting.")
                    use_pretrained = False  # Force training from scratch
            else:
                print("⚠️  Could not detect architecture. Training from scratch.")
                use_pretrained = False
        except Exception as e:
            print(f"⚠️  Warning: Could not load pre-trained model: {e}. Training from scratch.")
            use_pretrained = False
    
    if not use_pretrained:
        # Train from scratch with new improved architecture
        model = None  # Will be created in train_model
        print("🚀 Training from scratch with improved architecture:")
        print(f"   - Reduced capacity: 1024→512→256→128 (was 1280→640→320→160)")
        print(f"   - Increased dropout: {config.dropout}")
        print(f"   - Stronger regularization: weight_decay={config.weight_decay}")
        print(f"   - Enhanced augmentation: prob={config.augment_prob}")
    else:
        # Fine-tune with lower learning rate for transfer learning
        config.learning_rate = config.learning_rate * 0.5  # Lower LR for fine-tuning
        config.epochs = min(config.epochs, 40)  # Fewer epochs for fine-tuning
        print(f"🔄 Fine-tuning pre-trained model with LR={config.learning_rate}")
    
    report = train_model(train_split, test_split, config=config, label_map=label_map, pretrained_model=model)
    scaler.save(base_dir / "artifacts" / "scaler.mean.npy")
    return report

