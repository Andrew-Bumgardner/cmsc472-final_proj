#!/usr/bin/env python3

import torch
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from train import BaseballEventDetector, Config
from file_manager import FileManager
from torch.utils.data import Dataset, DataLoader

class PredictionDataset(Dataset):
    def __init__(self, frames_dir, sequence_length=6):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length

        self.frame_files = sorted(list(self.frames_dir.glob('frame_*.npy')))

        self.sequences = []
        for i in range(0, len(self.frame_files) - sequence_length + 1):
            self.sequences.append(i)

        logging.info(f"Created dataset with {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        frames = []

        for i in range(start_idx, start_idx + self.sequence_length):
            frame = np.load(str(self.frame_files[i])).astype(np.float32)
            frames.append(frame)

        sequence = torch.from_numpy(np.stack(frames)).float().permute(1, 0, 2, 3)
        return sequence, start_idx

def generate_predictions(model_path: str, output_dir: str = "predictions"):
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    config = Config()
    detector = BaseballEventDetector(config)

    checkpoint = torch.load(model_path, map_location=config.device)
    detector.model.load_state_dict(checkpoint['model_state_dict'])
    detector.model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_manager = FileManager()

    for split in ['train', 'val', 'test']:
        split_dir = getattr(file_manager, f'{split}_dir')
        if not split_dir.exists():
            continue

        logger.info(f"Processing {split} split")

        for game_dir in split_dir.iterdir():
            if not game_dir.is_dir():
                continue

            game_id = game_dir.name
            logger.info(f"Processing game: {game_id}")

            frame_files = list(game_dir.glob('frame_*.npy'))
            if not frame_files:
                logger.warning(f"No frame files found for game {game_id}")
                continue

            dataset = PredictionDataset(
                game_dir,
                sequence_length=config.sequence_length
            )

            if len(dataset) == 0:
                logger.warning(f"No valid sequences found for game {game_id}")
                continue

            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )

            predictions = []
            with torch.no_grad():
                for sequences, start_indices in tqdm(dataloader, desc=f"Predicting {game_id}"):
                    sequences = sequences.to(config.device)
                    outputs = detector.model(sequences)

                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred_class = torch.max(probs, dim=1)

                    pred_class = pred_class.cpu().numpy()
                    confidence = confidence.cpu().numpy()
                    start_indices = start_indices.numpy()

                    for i, (pred, conf, start_idx) in enumerate(zip(pred_class, confidence, start_indices)):
                        if pred != 0:  # Skip 'no_event' predictions
                            predictions.append({
                                "frame_idx": int(start_idx),
                                "event_type": {
                                    1: "home_run",
                                    2: "out",
                                    3: "hit"
                                }[int(pred)],
                                "confidence": float(conf)
                            })

            predictions.sort(key=lambda x: x['frame_idx'])

            output_file = output_dir / f"{game_id}_predictions.json"
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)

            logger.info(f"Saved predictions to {output_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate predictions from trained model')
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='predictions',
                       help='Output directory for predictions')

    args = parser.parse_args()

    try:
        generate_predictions(args.model, args.output)
        print("Prediction generation complete!")

    except Exception as e:
        logging.error(f"Error generating predictions: {str(e)}")
        raise

if __name__ == "__main__":
    main()