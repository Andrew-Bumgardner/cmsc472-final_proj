#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, 
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.models.video as video_models
import torchvision.transforms as transforms
from tqdm import tqdm
import yaml

from file_manager import FileManager

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
    
    @staticmethod
    def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return (batch - mean) / std

class TemporalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.temp_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.temp_attn(x)
        return x * attn

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

class BaseballEventDataset(Dataset):
    def __init__(self, frames_dir, labels_file=None, sequence_length=16, 
                 buffer_size=256, min_valid_frames=8, training=False):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size
        self.min_valid_frames = min_valid_frames
        self.training = training
        self.frame_buffer = {}
        self.buffer_queue = []
        
        self.labels = {}
        if labels_file and Path(labels_file).exists():
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
        
        self.video_dirs = [d for d in self.frames_dir.iterdir() if d.is_dir()]
        self.sequences = self._create_valid_sequences()
        print(f"Created dataset with {len(self.sequences)} valid sequences")
        self._calculate_class_weights()

    def _is_valid_sequence(self, video_dir: Path, start_idx: int) -> bool:
        end_idx = start_idx + self.sequence_length
        valid_frames = 0
        for i in range(start_idx, end_idx):
            if (video_dir / f"frame_{i:06d}.npy").exists():
                valid_frames += 1
        return valid_frames >= self.min_valid_frames

    def _create_valid_sequences(self) -> List[Tuple[str, int]]:
        valid_sequences = []
        for video_dir in self.video_dirs:
            video_name = video_dir.name
            frame_files = sorted(list(Path(video_dir).glob('frame_*.npy')))
            
            if video_name in self.labels:
                event_frames = [int(idx) for idx, label in self.labels[video_name].items() 
                              if int(label) > 0]
                
                for event_frame in event_frames:
                    start_idx = max(0, event_frame - self.sequence_length // 2)
                    if self._is_valid_sequence(video_dir, start_idx):
                        valid_sequences.append((str(video_dir), start_idx))
                
                if self.training and frame_files:
                    max_frame = max(int(f.stem.split('_')[1]) for f in frame_files)
                    for start_idx in range(0, max_frame - self.sequence_length, 
                                         self.sequence_length // 2):
                        if not any(abs(start_idx - ef) < self.sequence_length 
                                 for ef in event_frames):
                            if self._is_valid_sequence(video_dir, start_idx):
                                valid_sequences.append((str(video_dir), start_idx))
        
        return valid_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.clear()

        video_dir, start_idx = self.sequences[idx]
        
        try:
            frames = []
            end_idx = start_idx + self.sequence_length
            
            for i in range(start_idx, end_idx):
                frame_path = Path(video_dir) / f"frame_{i:06d}.npy"
                if frame_path.exists():
                    frame = np.load(str(frame_path)).astype(np.float32)
                else:
                    frame = frames[-1].copy() if frames else np.zeros((3, 224, 224), dtype=np.float32)
                frames.append(frame)
            
            sequence = np.stack(frames, axis=0)
            sequence = torch.from_numpy(sequence).float()
            sequence = sequence.permute(1, 0, 2, 3)
            
            video_name = Path(video_dir).name
            label = 0
            if video_name in self.labels:
                for i in range(start_idx, end_idx):
                    if str(i) in self.labels[video_name]:
                        label = int(self.labels[video_name][str(i)])
                        break
            
            return sequence, label
            
        except Exception as e:
            logging.error(f"Error loading sequence at {video_dir}, start_idx {start_idx}: {str(e)}")
            return torch.zeros((3, self.sequence_length, 224, 224)), 0
    
    def _calculate_class_weights(self):
        class_counts = np.zeros(4)  # [no_event, home_run, out, hit]
        for video_dir in self.video_dirs:
            video_name = video_dir.name
            if video_name in self.labels:
                for label in self.labels[video_name].values():
                    class_counts[int(label)] += 1
        
        smoothing_factor = 0.1
        class_counts += smoothing_factor
        
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        self.class_weights = torch.FloatTensor(weights)

def custom_collate(batch):
    sequences, labels = zip(*batch)
    sequences = torch.stack(sequences)
    
    if isinstance(labels[0], tuple):
        labels1, labels2, lambdas = zip(*labels)
        return sequences, (torch.tensor(labels1), torch.tensor(labels2), torch.tensor(lambdas))
    else:
        return sequences, torch.tensor(labels)

class BaseballEventDetector:
    def __init__(self, config: Any):
        self.config = config
        self.device = config.device
        
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        self.use_amp = hasattr(torch.cuda, 'amp') and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            logging.info("Using mixed precision training")
        else:
            logging.info("Using full precision training")
        
        self.best_val_f1 = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'val_precision': [], 'val_recall': [],
            'val_f1': [], 'learning_rates': []
        }
    
    def _create_model(self) -> nn.Module:
        base_model = video_models.r3d_18(weights=None)

        base_model.stem[0] = nn.Conv3d(
            3, 64, 
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
            bias=False
        )
        
        base_model.layer2.add_module('temporal_attention1', TemporalAttention(128))
        base_model.layer3.add_module('temporal_attention2', TemporalAttention(256))
        base_model.layer4.add_module('temporal_attention3', TemporalAttention(512))
        
        base_model.fc = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.num_classes)
        )
        
        return base_model

    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler
    ) -> float:
        self.model.train()
        total_loss = 0.0
        grad_norms = []
        
        torch.cuda.empty_cache() 

        pbar = tqdm(train_loader, desc=f"Training", leave=True)
        running_loss = 0.0
        running_samples = 0
        
        for batch_idx, batch in enumerate(pbar):
            try:
                if isinstance(batch[1], tuple):
                    sequences, (labels1, labels2, lambda_param) = batch
                    sequences = sequences.to(self.device)
                    labels1 = labels1.to(self.device)
                    labels2 = labels2.to(self.device)
                    batch_size = sequences.size(0)
                    
                    if self.use_amp:
                        with autocast():
                            outputs = self.model(sequences)
                            loss = (lambda_param * criterion(outputs, labels1) +
                                (1 - lambda_param) * criterion(outputs, labels2))
                    else:
                        outputs = self.model(sequences)
                        loss = (lambda_param * criterion(outputs, labels1) +
                            (1 - lambda_param) * criterion(outputs, labels2))
                else:
                    sequences, labels = batch
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    batch_size = sequences.size(0)
                    
                    if self.use_amp:
                        with autocast():
                            outputs = self.model(sequences)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = self.model(sequences)
                        loss = criterion(outputs, labels)
                
                running_loss += loss.item() * batch_size
                running_samples += batch_size
                avg_loss = running_loss / running_samples
                
                loss = loss / self.config.gradient_accumulation_steps
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(optimizer)
                        
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        grad_norms.append(total_norm)
                        
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        grad_norms.append(total_norm)
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                    'grad_norm': f'{total_norm:.4f}' if grad_norms else 'N/A'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    print("\nGPU OOM, skipping batch")
                    continue
                raise e
        
        self.metrics_history.setdefault('gradient_norms', []).append(np.mean(grad_norms))
        
        return total_loss / len(train_loader)

    def train(
        self,
        train_dir: str,
        labels_file: str,
        val_dir: Optional[str] = None,
        val_labels: Optional[str] = None
    ) -> Dict:
        print("\nInitializing training...")
        print(f"Training directory: {train_dir}")
        print(f"Using device: {self.device}")
        
        train_dataset = BaseballEventDataset(
            train_dir, labels_file,
            self.config.sequence_length,
            training=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=True
        )
        
        if val_dir and val_labels:
            val_dataset = BaseballEventDataset(
                val_dir, val_labels,
                self.config.sequence_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                num_workers=4,
                pin_memory=True,
                collate_fn=custom_collate
            )
        
        print(f"\nTraining set size: {len(train_dataset)}")
        if val_loader:
            print(f"Validation set size: {len(val_dataset)}")
        
        criterion = FocalLoss(
            gamma=2.0,
            alpha=train_dataset.class_weights.to(self.device)
        )
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25
        )
        
        print("\nStarting training...")
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 30)
            
            train_loss = self._train_epoch(
                train_loader, criterion, optimizer, scheduler
            )
            
            if val_loader:
                print("\nValidating...")
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['val_loss']
                
                print("\nMetrics:")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val F1 (macro): {val_metrics['macro_metrics']['f1']:.4f}")
                print(f"Per-class F1 scores:")
                for class_name, metrics in val_metrics['per_class_metrics'].items():
                    print(f"  {class_name}: {metrics['f1']:.4f}")
                
                if val_loss < (self.best_val_loss - self.config.min_delta):
                    print("\nValidation loss improved! Saving model...")
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_model("best_model.pth")
                else:
                    self.patience_counter += 1
                    remaining_patience = self.config.early_stopping_patience - self.patience_counter
                    print(f"\nValidation loss did not improve. Patience: {remaining_patience} epochs remaining")
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print("\nEarly stopping triggered!")
                    break
                
                self.metrics_history['train_loss'].append(train_loss)
                self.metrics_history['val_loss'].append(val_loss)
                self.metrics_history['val_precision'].append(val_metrics['macro_metrics']['precision'])
                self.metrics_history['val_recall'].append(val_metrics['macro_metrics']['recall'])
                self.metrics_history['val_f1'].append(val_metrics['macro_metrics']['f1'])
                self.metrics_history['learning_rates'].append(scheduler.get_last_lr()[0])
                
                self.plot_metrics()
        
        print("\nTraining completed!")
        return self.metrics_history
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        criterion = FocalLoss()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        class_mapping = {
            0: "no_event",
            1: "home_run",
            2: "out",
            3: "hit"
        }
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                try:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(sequences)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    logging.error(f"Evaluation error: {str(e)}")
                    continue
        
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'macro_metrics': {},
            'per_class_metrics': {},
            'confusion_matrix': confusion_matrix(
                all_labels, all_preds,
                labels=list(class_mapping.keys())
            ).tolist()
        }
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        metrics['macro_metrics'] = {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        }
        
        for class_idx, class_name in class_mapping.items():
            precision = precision_score(
                all_labels == class_idx,
                all_preds == class_idx,
                zero_division=0
            )
            recall = recall_score(
                all_labels == class_idx,
                all_preds == class_idx,
                zero_division=0
            )
            f1 = f1_score(
                all_labels == class_idx,
                all_preds == class_idx,
                zero_division=0
            )
            
            metrics['per_class_metrics'][class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(np.sum(all_labels == class_idx))
            }
        
        unique_classes = np.unique(np.concatenate([all_labels, all_preds]))
        if len(unique_classes) > 1 and all_probs.shape[1] == len(class_mapping):
            try:
                metrics['roc_auc'] = {}
                for class_idx in unique_classes:
                    class_probs = all_probs[:, class_idx]
                    class_labels = (all_labels == class_idx).astype(int)
                    if np.any(class_labels) and not np.all(class_labels):
                        auc = roc_auc_score(class_labels, class_probs)
                        metrics['roc_auc'][class_mapping[class_idx]] = auc
            except ValueError as e:
                logging.warning(f"ROC AUC calculation failed: {str(e)}")
                metrics['roc_auc'] = None
        
        return metrics
    
    def plot_metrics(self, save_path: str = "training_metrics.png"):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics_history['train_loss'], label='Train Loss')
        plt.plot(self.metrics_history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.metrics_history['val_precision'], label='Precision')
        plt.plot(self.metrics_history['val_recall'], label='Recall')
        plt.plot(self.metrics_history['val_f1'], label='F1')
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def save_model(self, filename: str):
        path = Path(self.config.checkpoints_dir) / filename
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'metrics_history': self.metrics_history,
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        if self.use_amp:
            save_dict['scaler'] = self.scaler.state_dict()
        
        torch.save(save_dict, path)
        logging.info(f"Model saved to {path}")

class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.model_type = "r3d_18"
        self.sequence_length = 8
        self.num_classes = 4
        self.learning_rate = 0.0001
        self.batch_size = 4
        self.num_epochs = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints_dir = "checkpoints"
        self.early_stopping_patience = 8
        self.gradient_clip_val = 1.0
        self.weight_decay = 0.01
        self.min_delta = 1e-4
        self.gradient_accumulation_steps = 2
        
        self.lr_schedule = {
            'initial_lr': 0.0001,
            'min_lr': 1e-6,
            'warmup_epochs': 3
        }
        
        if config_path:
            self.load_config(config_path)
        
        Path(self.checkpoints_dir).mkdir(exist_ok=True)
        self.validate_config()

    def validate_config(self):
        assert self.sequence_length > 0
        assert self.num_classes > 1
        assert self.learning_rate > 0
        assert self.num_epochs > 0
        assert self.gradient_accumulation_steps > 0
        assert self.early_stopping_patience > 0
        assert self.gradient_clip_val > 0
        assert self.weight_decay >= 0

    def load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    logging.warning(f"Unknown configuration parameter: {key}")
                    
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            print("Using default configuration")

def main():
    parser = argparse.ArgumentParser(description='Train baseball event detector')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        config = Config(args.config)
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