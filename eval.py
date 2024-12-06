#!/usr/bin/env python3

import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from train import BaseballEventDetector, BaseballEventDataset, Config
from file_manager import FileManager

def evaluate_model(model_path: str, test_dir: str, test_labels: str):
    logging.info(f"Evaluating model from {model_path}")
    logging.info(f"Test data directory: {test_dir}")

    config = Config()
    detector = BaseballEventDetector(config)

    checkpoint = torch.load(model_path, map_location=config.device)
    detector.model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Model loaded successfully")

    test_dataset = BaseballEventDataset(
        test_dir, 
        test_labels, 
        config.sequence_length, 
        prefix="test"
    )

    if len(test_dataset) == 0:
        raise ValueError(f"No test data found in {test_dir}")

    logging.info(f"Found {len(test_dataset)} test sequences")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    detector.model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(config.device)
            outputs = detector.model(sequences)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    if not all_labels or not all_preds:
        raise ValueError("No predictions were made. Check if test data is loading correctly.")

    logging.info(f"Processed {len(all_labels)} test samples")

    class_names = ["No Event", "Home Run", "Out", "Hit"]
    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)

    logging.info(f"Unique labels in test set: {unique_labels}")
    logging.info(f"Unique predictions made: {unique_preds}")

    labels = sorted(list(set(unique_labels) | set(unique_preds)))

    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        labels=labels,
        zero_division=0
    )

    conf_matrix = confusion_matrix(
        all_labels, 
        all_preds,
        labels=labels
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='YlOrRd',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return report, conf_matrix

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    file_manager = FileManager()
    model_path = "best_model.pth"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        report, conf_matrix = evaluate_model(
            model_path=model_path,
            test_dir=str(file_manager.test_dir),
            test_labels=str(file_manager.get_labels_file('test'))
        )

        print("\nTest Set Evaluation Report:")
        print(report)

        print("\nConfusion Matrix saved as 'confusion_matrix.png'")

    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()