#!/usr/bin/env python3
"""
Targeted model fix based on the specific errors identified in model_evaluation.json
Focus on the most common misclassification patterns
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
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simple_prediction import extract_simple_features
from src.preprocessing import FeatureConfig
from src.model import FaultSenseCNN


class ConfusionAwareLoss(nn.Module):
    """
    Custom loss that penalizes the most common confusion patterns
    """
    def __init__(self, confusion_penalties=None):
        super().__init__()
        self.confusion_penalties = confusion_penalties or {}
        self.base_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, outputs, targets):
        base_losses = self.base_loss(outputs, targets)
        
        # Get predictions
        _, predicted = torch.max(outputs, 1)
        
        # Apply confusion penalties
        penalties = torch.ones_like(base_losses)
        for i in range(len(targets)):
            true_label = targets[i].item()
            pred_label = predicted[i].item()
            
            # Increase penalty for common misclassifications
            confusion_key = f"{true_label}_{pred_label}"
            if confusion_key in self.confusion_penalties:
                penalties[i] = self.confusion_penalties[confusion_key]
        
        return (base_losses * penalties).mean()


def analyze_evaluation_errors():
    """Analyze the evaluation results to identify patterns."""
    print("🔍 Analyzing evaluation errors...")
    
    eval_path = Path("reports/model_evaluation.json")
    if not eval_path.exists():
        print("❌ Evaluation file not found")
        return None
    
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    # Count error patterns
    error_patterns = {}
    correct_patterns = {}
    
    for result in eval_data['results']:
        expected = result['expected']
        predicted = result['predicted']
        confidence = result['confidence']
        
        if result['correct']:
            if expected not in correct_patterns:
                correct_patterns[expected] = []
            correct_patterns[expected].append(confidence)
        else:
            pattern = f"{expected} → {predicted}"
            if pattern not in error_patterns:
                error_patterns[pattern] = []
            error_patterns[pattern].append(confidence)
    
    print("🚨 Most common error patterns:")
    sorted_errors = sorted(error_patterns.items(), key=lambda x: len(x[1]), reverse=True)
    
    confusion_penalties = {}
    label_to_idx = {
        "electrical_fault": 0,
        "fluid_leak": 1,
        "mechanical_fault": 2,
        "normal_operation": 3
    }
    
    for pattern, confidences in sorted_errors[:5]:
        avg_conf = np.mean(confidences)
        count = len(confidences)
        print(f"   {pattern}: {count} errors (avg conf: {avg_conf:.3f})")
        
        # Create penalty mapping
        parts = pattern.split(' → ')
        if len(parts) == 2:
            true_idx = label_to_idx.get(parts[0])
            pred_idx = label_to_idx.get(parts[1])
            if true_idx is not None and pred_idx is not None:
                confusion_key = f"{true_idx}_{pred_idx}"
                # Higher penalty for more frequent errors with high confidence
                penalty = 1.0 + (count / 10) + (avg_conf * 0.5)
                confusion_penalties[confusion_key] = penalty
    
    print(f"📊 Confusion penalties: {confusion_penalties}")
    return confusion_penalties, error_patterns


def targeted_retrain():
    """Retrain with targeted improvements."""
    print("🎯 Targeted Model Retraining")
    print("=" * 50)
    
    # Analyze errors first
    confusion_penalties, error_patterns = analyze_evaluation_errors()
    if confusion_penalties is None:
        print("❌ Could not analyze errors")
        return False
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load data more strategically - focus on problem cases
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
    
    # Extract features with emphasis on problematic classes
    features_list = []
    labels_list = []
    config = FeatureConfig()
    
    # Get more samples from classes that are commonly misclassified
    problem_classes = ['electrical_fault']  # Most errors from evaluation
    samples_per_class = {
        'electrical_fault': 40,  # More samples for problem class
        'fluid_leak': 30,
        'mechanical_fault': 30,
        'normal_operation': 30
    }
    
    for label in unique_labels:
        target_samples = samples_per_class.get(label, 30)
        label_files = manifest_df[manifest_df['label'] == label].sample(
            n=min(target_samples, len(manifest_df[manifest_df['label'] == label])),
            random_state=42
        )
        
        print(f"📊 Processing {len(label_files)} samples for {label}")
        
        for _, row in label_files.iterrows():
            file_path = Path(row['filepath'])
            
            if file_path.exists():
                try:
                    features = extract_simple_features(file_path, config)
                    if features is not None and len(features) > 0:
                        features_list.append(features)
                        labels_list.append(label_to_idx[label])
                except Exception as e:
                    continue
    
    if len(features_list) < 50:
        print("❌ Not enough features extracted")
        return False
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"📊 Training data: {features.shape[0]} samples")
    print(f"   Class distribution: {Counter(labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
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
    
    # Create model with better regularization
    input_dim = X_train_scaled.shape[1]
    num_classes = len(label_to_idx)
    model = FaultSenseCNN(input_dim, num_classes, dropout=0.5).to(device)  # Higher dropout
    
    print(f"🧠 Model: input_dim={input_dim}, classes={num_classes}")
    
    # Use confusion-aware loss
    criterion = ConfusionAwareLoss(confusion_penalties)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)  # Lower LR, higher weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training with early stopping
    num_epochs = 80
    batch_size = 16
    best_f1 = 0
    patience = 15
    patience_counter = 0
    
    print("\n🏋️ Targeted training...")
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                _, predicted = torch.max(test_outputs.data, 1)
                
                # Calculate F1 score
                from sklearn.metrics import f1_score
                predicted_np = predicted.cpu().numpy()
                f1 = f1_score(y_test, predicted_np, average='macro')
                accuracy = 100 * (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
                
                print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss/num_batches:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.3f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), "models/faultsense_cnn.pt")
                    print(f"   ✅ New best F1: {f1:.3f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    # Load best model and final evaluation
    model.load_state_dict(torch.load("models/faultsense_cnn.pt"))
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
    
    print(f"\n🎯 Targeted Fix Results:")
    print(f"   Final Accuracy: {final_accuracy:.2f}%")
    print(f"   Macro F1: {report['macro avg']['f1-score']:.3f}")
    
    # Show improvement in problem areas
    print(f"\n📊 Per-class performance:")
    for label in label_names:
        metrics = report[label]
        print(f"   {label}:")
        print(f"     Precision: {metrics['precision']:.3f}")
        print(f"     Recall: {metrics['recall']:.3f}")
        print(f"     F1-Score: {metrics['f1-score']:.3f}")
    
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
        "feature_type": "simple_features_targeted",
        "model_type": "FaultSenseCNN_TargetedFix",
        "label_map": label_to_idx,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "confusion_penalties": confusion_penalties
    }
    
    with open("models/registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Targeted fix complete!")
    print(f"   Model saved: models/faultsense_cnn.pt")
    print(f"   Registry updated with targeted improvements")
    
    return final_accuracy >= 70


if __name__ == "__main__":
    try:
        success = targeted_retrain()
        if success:
            print("\n🎉 Targeted fix successful! Model should have better predictions for problem cases.")
        else:
            print("\n⚠️  Targeted fix completed but may need further improvements.")
    except Exception as e:
        print(f"❌ Targeted fix failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
