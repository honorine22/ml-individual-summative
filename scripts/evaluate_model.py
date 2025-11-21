#!/usr/bin/env python3
"""Comprehensive model evaluation on test set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

API_URL = "http://127.0.0.1:8000"


def evaluate_test_set(test_dir: Path = Path("data/test")):
    """Evaluate model on entire test set."""
    print("="*70)
    print("Comprehensive Model Evaluation")
    print("="*70)
    
    all_results = []
    correct = 0
    total = 0
    
    # Test each class
    for label_dir in sorted(test_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        
        expected_label = label_dir.name
        files = list(label_dir.glob("*.wav"))
        
        print(f"\n📁 Testing {expected_label} ({len(files)} files)")
        print("-" * 70)
        
        class_correct = 0
        class_total = 0
        
        for test_file in tqdm(files[:20], desc=f"  {expected_label}", leave=False):  # Limit to 20 per class
            try:
                with open(test_file, "rb") as f:
                    files_data = {"file": (test_file.name, f, "audio/wav")}
                    resp = requests.post(f"{API_URL}/predict", files=files_data, timeout=10)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    predicted = result['label']
                    confidence = result['confidence']
                    is_correct = predicted == expected_label
                    
                    all_results.append({
                        'file': test_file.name,
                        'expected': expected_label,
                        'predicted': predicted,
                        'confidence': confidence,
                        'correct': is_correct
                    })
                    
                    total += 1
                    class_total += 1
                    if is_correct:
                        correct += 1
                        class_correct += 1
                        
            except Exception as e:
                print(f"  ⚠️  Error with {test_file.name}: {e}")
                continue
        
        if class_total > 0:
            class_acc = class_correct / class_total * 100
            print(f"  ✅ {class_correct}/{class_total} correct ({class_acc:.1f}%)")
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    overall_acc = correct / total * 100 if total > 0 else 0
    print(f"Total Tested: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {overall_acc:.2f}%")
    
    # Per-class breakdown
    print("\n" + "-"*70)
    print("Per-Class Accuracy:")
    print("-"*70)
    for label in sorted(set(r['expected'] for r in all_results)):
        class_results = [r for r in all_results if r['expected'] == label]
        class_correct = sum(1 for r in class_results if r['correct'])
        class_total = len(class_results)
        if class_total > 0:
            print(f"  {label:25s}: {class_correct:3d}/{class_total:3d} ({class_correct/class_total*100:5.1f}%)")
    
    # Confidence analysis
    print("\n" + "-"*70)
    print("Confidence Analysis:")
    print("-"*70)
    correct_high_conf = [r for r in all_results if r['correct'] and r['confidence'] > 0.7]
    incorrect_high_conf = [r for r in all_results if not r['correct'] and r['confidence'] > 0.7]
    correct_low_conf = [r for r in all_results if r['correct'] and r['confidence'] <= 0.7]
    incorrect_low_conf = [r for r in all_results if not r['correct'] and r['confidence'] <= 0.7]
    
    print(f"High confidence (>70%) correct:   {len(correct_high_conf):3d}")
    print(f"High confidence (>70%) incorrect: {len(incorrect_high_conf):3d}")
    print(f"Low confidence (≤70%) correct:     {len(correct_low_conf):3d}")
    print(f"Low confidence (≤70%) incorrect:  {len(incorrect_low_conf):3d}")
    
    if len(correct_high_conf) + len(incorrect_high_conf) > 0:
        high_conf_acc = len(correct_high_conf) / (len(correct_high_conf) + len(incorrect_high_conf)) * 100
        print(f"\nHigh confidence accuracy: {high_conf_acc:.1f}%")
    
    # Save detailed results
    results_file = Path("reports/model_evaluation.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump({
            'summary': {
                'total': total,
                'correct': correct,
                'accuracy': overall_acc
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✅ Detailed results saved to {results_file}")
    
    return overall_acc, all_results


if __name__ == "__main__":
    test_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/test")
    evaluate_test_set(test_dir)

