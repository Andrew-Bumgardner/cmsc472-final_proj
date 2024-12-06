#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    precision_score, 
    recall_score,
    f1_score,
    accuracy_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.optim import lr_scheduler

from file_manager import FileManager

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda':
            self.sequence_length = 3
            self.batch_size = 4
            self.num_workers = 4
            self.pin_memory = True
            self.num_epochs = 15
        else:
            self.sequence_length = 2
            self.batch_size = 2
            self.num_workers = 0
            self.pin_memory = False
            self.num_epochs = 1

        self.num_classes = 4
        self.learning_rate = 0.0005
        self.early_stopping_patience = 20

        logging.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

class VideoTransforms:
    @staticmethod
    def to_tensor(x: np.ndarray) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x

        x = torch.from_numpy(x)
        if x.shape[1] == 3:
            x = x.permute(1, 0, 2, 3)
        return x

    @staticmethod
    def to_float(x: torch.Tensor) -> torch.Tensor:
        return x.float()

    @staticmethod
    def get_train_transforms() -> transforms.Compose:
        return transforms.Compose([
            transforms.Lambda(VideoTransforms.to_tensor),
            transforms.Lambda(VideoTransforms.to_float),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2)
            ], p=0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transforms() -> transforms.Compose:
        return transforms.Compose([
            transforms.Lambda(VideoTransforms.to_tensor),
            transforms.Lambda(VideoTransforms.to_float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

class BaseballEventDataset(Dataset):
    def __init__(self, frames_dir, labels_file=None, sequence_length=16, training=False, prefix=""):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.training = training

        self.labels = {}
        if labels_file and Path(labels_file).exists():
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)

        self.sequences = self._create_valid_sequences()
        logging.info(f"{prefix}: Created dataset with {len(self.sequences)} valid sequences")

    def _create_valid_sequences(self) -> List[Tuple[str, int]]:
        valid_sequences = []
        video_dirs = [d for d in self.frames_dir.iterdir() if d.is_dir()]

        for video_dir in video_dirs:
            video_name = video_dir.name
            frame_files = sorted(list(Path(video_dir).glob('frame_*.npy')))

            if video_name in self.labels:
                event_frames = [int(idx) for idx, label in self.labels[video_name].items() 
                              if int(label) > 0]

                for event_frame in event_frames:
                    start_idx = max(0, event_frame - self.sequence_length // 2)
                    valid_sequences.append((str(video_dir), start_idx))

                if self.training and frame_files:
                    max_frame = max(int(f.stem.split('_')[1]) for f in frame_files)
                    for start_idx in range(0, max_frame - self.sequence_length, 
                                         self.sequence_length // 2):
                        if not any(abs(start_idx - ef) < self.sequence_length 
                                 for ef in event_frames):
                            valid_sequences.append((str(video_dir), start_idx))

        return valid_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_dir, start_idx = self.sequences[idx]

        frames = []
        end_idx = start_idx + self.sequence_length

        for i in range(start_idx, end_idx):
            frame_path = Path(video_dir) / f"frame_{i:06d}.npy"
            if frame_path.exists():
                frame = np.load(str(frame_path)).astype(np.float32)
            else:
                frame = frames[-1].copy() if frames else np.zeros((3, 224, 224), dtype=np.float32)
            frames.append(frame)

        sequence = torch.from_numpy(np.stack(frames)).float().permute(1, 0, 2, 3)

        video_name = Path(video_dir).name
        label = 0
        if video_name in self.labels:
            for i in range(start_idx, end_idx):
                if str(i) in self.labels[video_name]:
                    label = int(self.labels[video_name][str(i)])
                    break

        return sequence, label

class TrainingVisualizer:
    def __init__(self, save_dir: str = "training_plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'learning_rates': []
        }

        plt.style.use('bmh')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['lines.linewidth'] = 2

    def update_metrics(self, epoch_metrics: Dict, lr: Optional[float] = None):
        self.metrics['train_loss'].append(epoch_metrics.get('train_loss', None))
        self.metrics['val_loss'].append(epoch_metrics.get('val_loss', None))
        self.metrics['accuracy'].append(epoch_metrics.get('accuracy', None))

        if 'macro_metrics' in epoch_metrics:
            self.metrics['precision'].append(epoch_metrics['macro_metrics'].get('precision', None))
            self.metrics['recall'].append(epoch_metrics['macro_metrics'].get('recall', None))
            self.metrics['f1'].append(epoch_metrics['macro_metrics'].get('f1', None))

        if lr is not None:
            self.metrics['learning_rates'].append(lr)

    def plot_losses(self):
        plt.figure()
        epochs = range(1, len(self.metrics['train_loss']) + 1)

        plt.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss')
        if any(v is not None for v in self.metrics['val_loss']):
            plt.plot(epochs, self.metrics['val_loss'], 'r-', label='Validation Loss')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.savefig(self.save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metrics(self):
        plt.figure()
        epochs = range(1, len(self.metrics['accuracy']) + 1)

        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f']
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']

        for metric, color in zip(metrics_to_plot, colors):
            if any(v is not None for v in self.metrics[metric]):
                plt.plot(epochs, self.metrics[metric], color=color, label=metric.capitalize())

        plt.title('Training Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        plt.savefig(self.save_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_learning_rate(self):
        if not self.metrics['learning_rates']:
            return

        plt.figure()
        plt.plot(self.metrics['learning_rates'], color='#16a085')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)

        plt.savefig(self.save_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, conf_matrix: np.ndarray, class_names: List[str]):
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self):
        clean_metrics = {}
        for key, value in self.metrics.items():
            clean_metrics[key] = [v for v in value if v is not None]

        metrics_file = self.save_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(clean_metrics, f, indent=2)

    def generate_training_report(self):
        report = ["# Training Report\n"]

        report.append("## Training Summary")
        report.append(f"Total Epochs: {len(self.metrics['train_loss'])}")

        best_val_loss = min((v for v in self.metrics['val_loss'] if v is not None), default=None)
        best_f1 = max((v for v in self.metrics['f1'] if v is not None), default=None)
        best_accuracy = max((v for v in self.metrics['accuracy'] if v is not None), default=None)

        report.append("\n## Best Metrics")
        if best_val_loss is not None:
            report.append(f"- Best Validation Loss: {best_val_loss:.4f}")
        if best_f1 is not None:
            report.append(f"- Best F1 Score: {best_f1:.4f}")
        if best_accuracy is not None:
            report.append(f"- Best Accuracy: {best_accuracy:.4f}")

        report.append("\n## Generated Plots")
        report.append("- Loss Curves: `training_plots/loss_curves.png`")
        report.append("- Performance Metrics: `training_plots/performance_metrics.png`")
        report.append("- Learning Rate Schedule: `training_plots/learning_rate.png`")
        report.append("- Confusion Matrix: `training_plots/confusion_matrix.png`")

        report_file = self.save_dir / 'training_report.md'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))

class BaseballEventModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = video_models.r3d_18(weights=None)

        self.backbone.stem[0] = nn.Conv3d(
            3, 64,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(4, 3, 3),
            bias=False
        )

        self.motion_branch = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.stem(x)
        x = x + self.motion_branch(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.weights = torch.tensor([0.2, 8.0, 3.0, 2.0])

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.weights = self.weights.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        weight = self.weights[targets]
        return (weight * focal_loss).mean()

class BaseballEventDetector:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.model = BaseballEventModel(num_classes=config.num_classes).to(self.device)
        self.best_f1_score = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        if self.device.type == 'cuda':
            logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = {
            'val_loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'macro_metrics': {
                'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
                'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
                'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0)
            }
        }

        return metrics

    def save_model(self, filename: str):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss,
            'best_f1_score': self.best_f1_score
        }
        torch.save(save_dict, filename)
        logging.info(f"Model saved to {filename}")

    def _train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0.0

        for sequences, labels in tqdm(train_loader, desc="Training"):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def train(self, train_dir, labels_file, val_dir=None, val_labels=None):
        device_name = "GPU" if self.device.type == "cuda" else "CPU"
        logging.info(f"Starting training on {device_name}...")
        logging.info(f"Training data: {train_dir}")

        visualizer = TrainingVisualizer()

        train_loader = DataLoader(
            BaseballEventDataset(train_dir, labels_file, self.config.sequence_length, training=True, prefix="train"),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        val_loader = None
        if val_dir and val_labels:
            val_loader = DataLoader(
                BaseballEventDataset(val_dir, val_labels, self.config.sequence_length, prefix="validation"),
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )

        criterion = (FocalLoss().to(self.device) if self.device.type == 'cuda' 
                    else nn.CrossEntropyLoss())

        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.0
        )

        if self.device.type == 'cuda':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate * 10,
                epochs=self.config.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.2,
                div_factor=20,
                final_div_factor=1000
            )
        else:
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )

        try:
            for epoch in range(self.config.num_epochs):
                logging.info(f"Epoch {epoch+1}/{self.config.num_epochs}")

                train_loss = self._train_epoch(train_loader, criterion, optimizer)
                logging.info(f"Training loss: {train_loss:.4f}")

                current_lr = optimizer.param_groups[0]['lr']

                epoch_metrics = {'train_loss': train_loss}

                if val_loader:
                    val_metrics = self.evaluate(val_loader)
                    epoch_metrics.update(val_metrics)

                    logging.info(f"Validation loss: {val_metrics['val_loss']:.4f}")
                    logging.info(f"Validation F1: {val_metrics['macro_metrics']['f1']:.4f}")

                    if self.device.type == 'cuda':
                        scheduler.step()
                    else:
                        scheduler.step(val_metrics['val_loss'])

                    if val_metrics['macro_metrics']['f1'] > self.best_f1_score:
                        logging.info("New best model, saving...")
                        self.best_f1_score = val_metrics['macro_metrics']['f1']
                        self.patience_counter = 0
                        self.save_model("best_model.pth")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.early_stopping_patience:
                            logging.warning("Early stopping triggered!")
                            break

                visualizer.update_metrics(epoch_metrics, current_lr)
                visualizer.plot_losses()
                visualizer.plot_metrics()
                visualizer.plot_learning_rate()

            visualizer.save_metrics()
            visualizer.generate_training_report()

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            visualizer.save_metrics()
            visualizer.generate_training_report()
            if train_loss < self.best_val_loss:
                self.save_model("best_model.pth")
                logging.info("Model saved before interruption")

        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        logging.info("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train baseball event detector')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )

    try:
        config = Config()
        detector = BaseballEventDetector(config)

        file_manager = FileManager()

        detector.train(
            train_dir=str(file_manager.train_dir),
            labels_file=str(file_manager.get_labels_file('train')),
            val_dir=str(file_manager.val_dir),
            val_labels=str(file_manager.get_labels_file('val'))
        )

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(0)

    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()