"""CLI orchestration for data download, training, evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model import TrainConfig, train_model
from src.preprocessing import prepare_dataset, warm_wav2vec_cache


def parse_args():
    parser = argparse.ArgumentParser(description="FaultSense pipeline runner")
    parser.add_argument("--stage", choices=["download", "train"], required=True)
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mlflow-experiment", default="FaultSense", help="MLflow experiment name")
    return parser.parse_args()


def stage_download(data_dir: Path):
    warm_wav2vec_cache()
    _, _, _ = prepare_dataset(data_dir)
    print("Dataset downloaded and curated.")


def stage_train(data_dir: Path, args):
    warm_wav2vec_cache()
    train_split, test_split, _ = prepare_dataset(data_dir)
    label_map_path = data_dir / "artifacts" / "label_to_idx.json"
    label_map = json.loads(label_map_path.read_text()) if label_map_path.exists() else {}
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mlflow_experiment=args.mlflow_experiment,
    )
    report = train_model(train_split, test_split, config=config, label_map=label_map)
    print(report.to_json())


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if args.stage == "download":
        stage_download(data_dir)
    elif args.stage == "train":
        stage_train(data_dir, args)


if __name__ == "__main__":
    main()

