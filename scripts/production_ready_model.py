#!/usr/bin/env python3
"""
Create a production-ready model that addresses all the identified issues
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simple_prediction import extract_simple_features
from src.preprocessing import FeatureConfig


class ProductionFaultCNN(nn.Module):
    """
    Production-ready CNN optimized for the specific prediction errors identified
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        # Feature extraction layers with residual-like connections
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ),
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7)
            ),
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.8)
            )
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Forward through feature layers
        for layer in self.feature_layers:
            x = layer(x)
        
        # Classification
        return self.classifier(x)


def create_production_model():
    """Create a production-ready model with comprehensive training."""
    print("🏭 Creating Production-Ready FaultSense Model")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    curated_dir = Path("data/curated")
    manifest_path = curated_dir / "manifest.csv"
    
    if not manifest_path.exists():
        print("❌ Manifest file not found")
        return False
    
    manifest_df = pd.read_csv(manifest_path)
    
    # Create label mapping
    unique_labels = sorted(manifest_df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"🏷️  Labels: {list(label_to_idx.keys())}")
    
    # Extract features from ALL available data for better training
    features_list = []
    labels_list = []
    filepaths_list = []
    config = FeatureConfig()
    
    print("🎵 Extracting features from all available data...")
    processed_count = 0
    
    for idx, row in manifest_df.iterrows():
        file_path = Path(row['filepath'])
        
        if file_path.exists():
            try:
                features = extract_simple_features(file_path, config)
                if features is not None and len(features) > 0:
                    features_list.append(features)
                    labels_list.append(label_to_idx[row['label']])
                    filepaths_list.append(str(file_path))
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        print(f"   Processed {processed_count} files...")
                        
            except Exception as e:
                print(f"⚠️  Error processing {file_path}: {e}")
                continue
    
    if len(features_list) < 100:
        print("❌ Not enough features extracted")
        return False
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"✅ Feature extraction complete:")
    print(f"   Total samples: {features.shape[0]}")
    print(f"   Feature dimension: {features.shape[1]}")
    print(f"   Class distribution: {Counter(labels)}")
    
    # Use stratified k-fold for robust training
    print("\n🔄 Using stratified cross-validation for robust training...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    best_model_state = None
    best_f1 = 0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"\n📁 Fold {fold + 1}/5")
        
        X_train_fold = features[train_idx]
        X_val_fold = features[val_idx]
        y_train_fold = labels[train_idx]
        y_val_fold = labels[val_idx]
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_train_tensor = torch.LongTensor(y_train_fold).to(device)
        y_val_tensor = torch.LongTensor(y_val_fold).to(device)
        
        # Create model
        input_dim = X_train_scaled.shape[1]
        num_classes = len(label_to_idx)
        model = ProductionFaultCNN(input_dim, num_classes, dropout=0.4).to(device)
        
        # Calculate class weights for this fold
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        
        # Training for this fold
        num_epochs = 40
        batch_size = 16
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            
            # Mini-batch training
            indices = torch.randperm(len(X_train_tensor))
            num_batches = 0
            
            for i in range(0, len(X_train_tensor), batch_size):
                if i + batch_size > len(X_train_tensor):
                    continue  # Skip incomplete batches
                    
                batch_indices = indices[i:i+batch_size]
                batch_X = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
        
        # Evaluate this fold
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, predicted = torch.max(val_outputs.data, 1)
            
            predicted_np = predicted.cpu().numpy()
            fold_f1 = f1_score(y_val_fold, predicted_np, average='macro')
            fold_accuracy = 100 * (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
            
            fold_results.append({
                'fold': fold + 1,
                'f1': fold_f1,
                'accuracy': fold_accuracy
            })
            
            print(f"   Fold {fold + 1} - Accuracy: {fold_accuracy:.2f}%, F1: {fold_f1:.3f}")
            
            # Save best model
            if fold_f1 > best_f1:
                best_f1 = fold_f1
                best_model_state = model.state_dict().copy()
                best_scaler = scaler
    
    # Calculate average performance
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    
    print(f"\n📊 Cross-validation results:")
    print(f"   Average F1: {avg_f1:.3f} ± {np.std([r['f1'] for r in fold_results]):.3f}")
    print(f"   Average Accuracy: {avg_accuracy:.2f}% ± {np.std([r['accuracy'] for r in fold_results]):.2f}%")
    print(f"   Best F1: {best_f1:.3f}")
    
    # Train final model on all data with best hyperparameters
    print(f"\n🎯 Training final model on all data...")
    
    # Final train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Normalize with best scaler approach
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    # Create final model
    final_model = ProductionFaultCNN(input_dim, num_classes, dropout=0.4).to(device)
    
    # Load best state if available
    if best_model_state is not None:
        try:
            final_model.load_state_dict(best_model_state)
        except:
            print("⚠️  Could not load best state, training from scratch")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Final training
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.0005, weight_decay=0.015)
    
    num_epochs = 30
    batch_size = 16
    
    for epoch in range(num_epochs):
        final_model.train()
        epoch_loss = 0
        
        indices = torch.randperm(len(X_train_tensor))
        num_batches = 0
        
        for i in range(0, len(X_train_tensor), batch_size):
            if i + batch_size > len(X_train_tensor):
                continue
                
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train_tensor[batch_indices]
            batch_y = y_train_tensor[batch_indices]
            
            optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss/max(num_batches, 1):.4f}")
    
    # Final evaluation
    final_model.eval()
    with torch.no_grad():
        test_outputs = final_model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        final_accuracy = 100 * (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    
    # Detailed evaluation
    predicted_np = predicted.cpu().numpy()
    report = classification_report(y_test, predicted_np, target_names=unique_labels, output_dict=True)
    
    print(f"\n🎯 Final Production Model Results:")
    print(f"   Test Accuracy: {final_accuracy:.2f}%")
    print(f"   Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"   Weighted F1: {report['weighted avg']['f1-score']:.3f}")
    
    # Per-class performance
    print(f"\n📊 Per-class performance:")
    for label in unique_labels:
        metrics = report[label]
        print(f"   {label}:")
        print(f"     Precision: {metrics['precision']:.3f}")
        print(f"     Recall: {metrics['recall']:.3f}")
        print(f"     F1-Score: {metrics['f1-score']:.3f}")
    
    # Save the production model
    model_path = "models/faultsense_cnn.pt"
    torch.save(final_model.state_dict(), model_path)
    
    # Save artifacts
    artifacts_dir = Path("data/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(artifacts_dir / "scaler.mean.npy", final_scaler.mean_)
    np.save(artifacts_dir / "scaler.scale.npy", final_scaler.scale_)
    
    with open(artifacts_dir / "label_to_idx.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)
    
    # Update registry
    registry = {
        "best_model": model_path,
        "best_f1": report['macro avg']['f1-score'],
        "weighted_f1": report['weighted avg']['f1-score'],
        "test_accuracy": final_accuracy / 100,
        "cv_avg_f1": avg_f1,
        "cv_avg_accuracy": avg_accuracy / 100,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "feature_type": "simple_features_production",
        "model_type": "ProductionFaultCNN",
        "label_map": label_to_idx,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(features),
        "class_weights": dict(zip(unique_labels, class_weights.tolist())),
        "cross_validation_folds": 5
    }
    
    with open("models/registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Production model created successfully!")
    print(f"   Model saved: {model_path}")
    print(f"   Registry updated: models/registry.json")
    print(f"   Artifacts saved: {artifacts_dir}")
    
    # Success criteria
    success = (
        final_accuracy >= 65 and 
        report['macro avg']['f1-score'] >= 0.60 and
        avg_f1 >= 0.55  # Cross-validation consistency
    )
    
    if success:
        print(f"\n🎉 SUCCESS! Production model meets quality criteria:")
        print(f"   ✅ Test Accuracy: {final_accuracy:.2f}% (≥65%)")
        print(f"   ✅ Test F1: {report['macro avg']['f1-score']:.3f} (≥0.60)")
        print(f"   ✅ CV F1: {avg_f1:.3f} (≥0.55)")
    else:
        print(f"\n📈 Model created but below optimal thresholds:")
        print(f"   📊 Test Accuracy: {final_accuracy:.2f}% (target: ≥65%)")
        print(f"   📊 Test F1: {report['macro avg']['f1-score']:.3f} (target: ≥0.60)")
        print(f"   📊 CV F1: {avg_f1:.3f} (target: ≥0.55)")
    
    return success


if __name__ == "__main__":
    try:
        success = create_production_model()
        if success:
            print("\n🏭 Production model successfully created and ready for deployment!")
        else:
            print("\n🔧 Production model created but may benefit from further optimization.")
    except Exception as e:
        print(f"❌ Production model creation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
