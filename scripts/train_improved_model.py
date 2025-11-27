#!/usr/bin/env python3
"""
Train an improved FaultSense model with better architecture and training strategies
Addresses the performance issues identified in model evaluation
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simple_prediction import extract_simple_features
from src.preprocessing import FeatureConfig


class ImprovedFaultSenseCNN(nn.Module):
    """
    Improved CNN architecture with:
    - Better feature extraction layers
    - Attention mechanism for important features
    - Residual connections
    - Advanced regularization
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        # Feature extraction with residual blocks
        self.feature_extractor = nn.Sequential(
            # First block - wide receptive field
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            # Second block with residual connection
            nn.Linear(2048, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            
            # Third block - feature compression
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )
        
        # Classification head with multiple paths
        self.classifier_main = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary classifier for better gradient flow
        self.classifier_aux = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, return_aux=False):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Main classification
        main_output = self.classifier_main(attended_features)
        
        if return_aux and self.training:
            # Auxiliary output for training
            aux_output = self.classifier_aux(features)
            return main_output, aux_output
        
        return main_output


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def extract_features_from_curated_data():
    """Extract features from curated audio data with improved preprocessing."""
    print("🎵 Extracting features from curated audio data...")
    
    curated_dir = Path("data/curated")
    if not curated_dir.exists():
        raise FileNotFoundError(f"Curated data directory not found: {curated_dir}")
    
    # Load manifest
    manifest_path = curated_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    import pandas as pd
    manifest_df = pd.read_csv(manifest_path)
    print(f"📊 Found {len(manifest_df)} audio files in manifest")
    
    features_list = []
    labels_list = []
    filepaths_list = []
    
    # Create label mapping
    unique_labels = sorted(manifest_df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"🏷️  Labels: {list(label_to_idx.keys())}")
    
    # Process each audio file
    for idx, row in manifest_df.iterrows():
        # Use the full filepath from manifest
        file_path = Path(row['filepath'])
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
            
        try:
            # Extract features using improved method
            config = FeatureConfig()
            features = extract_simple_features(file_path, config)
            
            if features is not None and len(features) > 0:
                features_list.append(features)
                labels_list.append(label_to_idx[row['label']])
                filepaths_list.append(str(file_path))
                
                if len(features_list) % 50 == 0:
                    print(f"   Processed {len(features_list)} files...")
            else:
                print(f"⚠️  Failed to extract features from: {file_path}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    if not features_list:
        raise ValueError("No features extracted from audio files")
    
    # Convert to arrays
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    print(f"✅ Feature extraction complete:")
    print(f"   Features shape: {features_array.shape}")
    print(f"   Labels shape: {labels_array.shape}")
    print(f"   Feature dimension: {features_array.shape[1]}")
    
    return features_array, labels_array, filepaths_list, label_to_idx


def train_improved_model():
    """Train the improved model with advanced techniques."""
    print("🚀 Training Improved FaultSense Model")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Extract features
    features, labels, filepaths, label_map = extract_features_from_curated_data()
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"📊 Data splits:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples") 
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create model
    input_dim = X_train_scaled.shape[1]
    num_classes = len(label_map)
    model = ImprovedFaultSenseCNN(input_dim, num_classes, dropout=0.3).to(device)
    
    print(f"🧠 Model architecture:")
    print(f"   Input dimension: {input_dim}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion_main = FocalLoss(alpha=1, gamma=2)
    criterion_aux = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training parameters
    num_epochs = 100
    batch_size = 32
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    # Training loop
    print("\n🏋️ Starting training...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        # Mini-batch training
        num_batches = (len(X_train_tensor) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X_train_tensor))
            
            batch_X = X_train_tensor[start_idx:end_idx]
            batch_y = y_train_tensor[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # Forward pass with auxiliary output
            main_output, aux_output = model(batch_X, return_aux=True)
            
            # Combined loss
            main_loss = criterion_main(main_output, batch_y)
            aux_loss = criterion_aux(aux_output, batch_y)
            total_loss = main_loss + 0.3 * aux_loss  # Weight auxiliary loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion_main(val_output, y_val_tensor).item()
            
            _, predicted = torch.max(val_output.data, 1)
            total = y_val_tensor.size(0)
            correct = (predicted == y_val_tensor).sum().item()
            val_accuracy = 100 * correct / total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record metrics
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "models/best_improved_model.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load("models/best_improved_model.pt"))
    model.eval()
    
    # Final test evaluation
    with torch.no_grad():
        test_output = model(X_test_tensor)
        _, test_predicted = torch.max(test_output.data, 1)
        test_accuracy = 100 * (test_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    
    print(f"\n🎯 Final Results:")
    print(f"   Test Accuracy: {test_accuracy:.2f}%")
    
    # Detailed evaluation
    test_predicted_np = test_predicted.cpu().numpy()
    y_test_np = y_test
    
    # Classification report
    label_names = list(label_map.keys())
    report = classification_report(y_test_np, test_predicted_np, 
                                 target_names=label_names, output_dict=True)
    
    print(f"\n📊 Detailed Performance:")
    for label in label_names:
        metrics = report[label]
        print(f"   {label}:")
        print(f"     Precision: {metrics['precision']:.3f}")
        print(f"     Recall: {metrics['recall']:.3f}")
        print(f"     F1-Score: {metrics['f1-score']:.3f}")
    
    # Save final model
    final_model_path = "models/faultsense_cnn.pt"
    torch.save(model.state_dict(), final_model_path)
    
    # Save scaler
    artifacts_dir = Path("data/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_
    np.save(artifacts_dir / "scaler.mean.npy", scaler_mean)
    np.save(artifacts_dir / "scaler.scale.npy", scaler_scale)
    
    # Save label mapping
    with open(artifacts_dir / "label_to_idx.json", "w") as f:
        json.dump(label_map, f, indent=2)
    
    # Update registry
    registry = {
        "best_model": final_model_path,
        "best_f1": report['macro avg']['f1-score'],
        "test_accuracy": test_accuracy / 100,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "feature_type": "improved_simple_features",
        "model_type": "ImprovedFaultSenseCNN",
        "label_map": label_map,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test)
    }
    
    with open("models/registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Model training complete!")
    print(f"   Model saved: {final_model_path}")
    print(f"   Registry updated: models/registry.json")
    print(f"   Artifacts saved: {artifacts_dir}")
    
    return model, test_accuracy, report


if __name__ == "__main__":
    try:
        model, accuracy, report = train_improved_model()
        print(f"\n🎉 Training successful! Test accuracy: {accuracy:.2f}%")
        
        if accuracy >= 75:
            print("✅ Target accuracy (≥75%) achieved!")
        else:
            print("⚠️  Target accuracy not yet reached. Consider further improvements.")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
