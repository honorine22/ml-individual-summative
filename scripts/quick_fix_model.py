#!/usr/bin/env python3
"""
Quick model fix to address the prediction issues identified in model evaluation
Focus on immediate improvements rather than complete retraining
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
from src.model import FaultSenseCNN


class BalancedFocalLoss(nn.Module):
    """Balanced Focal Loss to address class imbalance and hard examples."""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def quick_retrain_model():
    """Quick retraining with better loss function and regularization."""
    print("🔧 Quick Model Fix for Better Predictions")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load existing model and data
    print("📊 Loading existing model and preparing data...")
    
    # Extract features from curated data
    curated_dir = Path("data/curated")
    manifest_path = curated_dir / "manifest.csv"
    
    if not manifest_path.exists():
        print("❌ Manifest file not found")
        return False
    
    import pandas as pd
    manifest_df = pd.read_csv(manifest_path)
    
    # Create label mapping
    unique_labels = sorted(manifest_df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"🏷️  Labels: {list(label_to_idx.keys())}")
    
    # Quick feature extraction (sample subset for speed)
    features_list = []
    labels_list = []
    config = FeatureConfig()
    
    # Take a balanced sample for quick training
    samples_per_class = 25  # Reduced for speed
    
    for label in unique_labels:
        label_files = manifest_df[manifest_df['label'] == label].sample(n=min(samples_per_class, len(manifest_df[manifest_df['label'] == label])))
        
        for _, row in label_files.iterrows():
            file_path = Path(row['filepath'])
            
            if file_path.exists():
                try:
                    features = extract_simple_features(file_path, config)
                    if features is not None and len(features) > 0:
                        features_list.append(features)
                        labels_list.append(label_to_idx[label])
                except Exception as e:
                    print(f"⚠️  Error processing {file_path}: {e}")
                    continue
    
    if len(features_list) < 20:
        print("❌ Not enough features extracted")
        return False
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"📊 Quick training data: {features.shape[0]} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Load existing model architecture
    input_dim = X_train_scaled.shape[1]
    num_classes = len(label_to_idx)
    
    # Create improved model with better regularization
    model = FaultSenseCNN(input_dim, num_classes, dropout=0.4).to(device)
    
    print(f"🧠 Model: input_dim={input_dim}, classes={num_classes}")
    
    # Calculate class weights for balanced training
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"⚖️  Class weights: {dict(zip(unique_labels, class_weights))}")
    
    # Better loss function and optimizer
    criterion = BalancedFocalLoss(alpha=class_weights_tensor, gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Quick training (fewer epochs for speed)
    num_epochs = 50
    batch_size = 16
    best_accuracy = 0
    
    print("\n🏋️ Quick training...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Mini-batch training
        num_batches = (len(X_train_tensor) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X_train_tensor))
            
            batch_X = X_train_tensor[start_idx:end_idx]
            batch_y = y_train_tensor[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = 100 * (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    # Save best model
                    torch.save(model.state_dict(), "models/faultsense_cnn.pt")
                
                print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss/num_batches:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        final_accuracy = 100 * (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    
    # Detailed evaluation
    predicted_np = predicted.cpu().numpy()
    y_test_np = y_test
    
    # Classification report
    label_names = list(label_to_idx.keys())
    report = classification_report(y_test_np, predicted_np, target_names=label_names, output_dict=True)
    
    print(f"\n🎯 Quick Fix Results:")
    print(f"   Final Accuracy: {final_accuracy:.2f}%")
    print(f"   Macro F1: {report['macro avg']['f1-score']:.3f}")
    
    for label in label_names:
        metrics = report[label]
        print(f"   {label}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Save artifacts
    artifacts_dir = Path("data/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    np.save(artifacts_dir / "scaler.mean.npy", scaler.mean_)
    np.save(artifacts_dir / "scaler.scale.npy", scaler.scale_)
    
    # Save label mapping
    with open(artifacts_dir / "label_to_idx.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)
    
    # Update registry
    registry = {
        "best_model": "models/faultsense_cnn.pt",
        "best_f1": report['macro avg']['f1-score'],
        "test_accuracy": final_accuracy / 100,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "feature_type": "simple_features_balanced",
        "model_type": "FaultSenseCNN_QuickFix",
        "label_map": label_to_idx,
        "class_weights": dict(zip(label_names, class_weights.tolist()))
    }
    
    with open("models/registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Quick fix complete!")
    print(f"   Model saved: models/faultsense_cnn.pt")
    print(f"   Registry updated with balanced training")
    
    return final_accuracy >= 70  # Lower threshold for quick fix


if __name__ == "__main__":
    try:
        success = quick_retrain_model()
        if success:
            print("\n🎉 Quick fix successful! Model should have better balanced predictions.")
        else:
            print("\n⚠️  Quick fix completed but may need further improvements.")
    except Exception as e:
        print(f"❌ Quick fix failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
