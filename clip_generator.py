#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from threading import Lock

import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms

from train import BaseballEventModel

@dataclass
class ClipConfig:
    before_event: int = 60  # Number of frames before event
    after_event: int = 90   # Number of frames after event
    min_score: float = 1.0  # Minimum score threshold
    fade_duration: int = 15 # Frames for fade transitions
    fps: int = 30          # Output video framerate
    min_clip_gap: int = 120 # Minimum frames between clips
    target_size: tuple = (640, 480)  # Output video dimensions
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32   # Batch size for inference
    confidence_threshold: float = 0.8  # Minimum confidence for event detection

class EventDetector:
    def __init__(self, model_path: str, config: ClipConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.lock = Lock()

        self.event_types = ["no_event", "home_run", "out", "hit"]

        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = BaseballEventModel(num_classes=len(self.event_types))
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def detect_events(self, frames: np.ndarray) -> List[Dict]:
        with self.lock:
            events = []
            for i in range(0, len(frames), self.config.batch_size):
                batch_frames = frames[i:i + self.config.batch_size]
                batch_tensors = []

                for frame in batch_frames:
                    frame_tensor = self.transform(frame).unsqueeze(0)
                    batch_tensors.append(frame_tensor)

                batch = torch.cat(batch_tensors).to(self.device)

                with torch.no_grad():
                    outputs = self.model(batch)
                    probs = torch.softmax(outputs, dim=1)

                    for j, prob in enumerate(probs):
                        conf, pred = torch.max(prob, dim=0)
                        if conf.item() >= self.config.confidence_threshold and pred.item() != 0:
                            events.append({
                                'frame_idx': i + j,
                                'event_type': self.event_types[pred.item()],
                                'confidence': conf.item()
                            })

                del batch_tensors
                del batch

            torch.cuda.empty_cache()

            return self._merge_events(events)

    def _merge_events(self, events: List[Dict]) -> List[Dict]:
        if not events:
            return []

        merged = []
        current_event = events[0]
        current_frame = events[0]['frame_idx']

        for event in events[1:]:
            if (event['frame_idx'] - current_frame <= self.config.min_clip_gap and
                event['event_type'] == current_event['event_type']):
                if event['confidence'] > current_event['confidence']:
                    current_event = event
            else:
                merged.append(current_event)
                current_event = event
            current_frame = event['frame_idx']

        merged.append(current_event)
        return merged

class HighlightGenerator:
    def __init__(
        self,
        model_path: str,
        output_dir: str = "highlights",
        config: Optional[ClipConfig] = None
    ):
        self.config = config or ClipConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.detector = EventDetector(model_path, self.config)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_highlights(
        self,
        video_path: Union[str, Path],
        output_name: Optional[str] = None
    ) -> Optional[str]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_name is None:
            output_name = f"{video_path.stem}_highlights.mp4"

        output_path = self.output_dir / output_name

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = int(cap.get(cv2.CAP_PROP_FPS))

            self.logger.info("Reading video frames...")
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frames = np.array(frames)

            self.logger.info("Detecting events...")
            events = self.detector.detect_events(frames)

            if not events:
                self.logger.info("No significant events detected")
                return None

            self.logger.info("Generating highlight clips...")
            clips = self._create_clips(frames, events)

            if clips:
                self._combine_clips(clips, output_path)
                self.logger.info(f"Highlights saved to: {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            self.logger.error(f"Error generating highlights: {str(e)}")
            raise

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def _create_clips(
        self,
        frames: np.ndarray,
        events: List[Dict]
    ) -> List[np.ndarray]:
        clips = []

        for event in events:
            start_frame = max(0, event['frame_idx'] - self.config.before_event)
            end_frame = min(len(frames), 
                          event['frame_idx'] + self.config.after_event)

            clip = frames[start_frame:end_frame]
            clip = self._apply_fade_effects(clip)
            clip = self._add_event_overlay(clip, event)            
            clips.append(clip)

        return clips

    def _apply_fade_effects(self, clip: np.ndarray) -> np.ndarray:
        for i in range(self.config.fade_duration):
            alpha = i / self.config.fade_duration
            clip[i] = cv2.addWeighted(
                np.zeros_like(clip[i]), 1 - alpha,
                clip[i], alpha, 0
            )

            alpha = i / self.config.fade_duration
            clip[-(i+1)] = cv2.addWeighted(
                np.zeros_like(clip[-(i+1)]), alpha,
                clip[-(i+1)], 1 - alpha, 0
            )

        return clip

    def _add_event_overlay(
        self,
        clip: np.ndarray,
        event: Dict
    ) -> np.ndarray:
        for i in range(len(clip)):
            frame = clip[i].copy()

            overlay = np.zeros_like(frame)
            cv2.rectangle(
                overlay,
                (10, frame.shape[0] - 60),
                (300, frame.shape[0] - 10),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                overlay,
                f"{event['event_type'].upper()}",
                (20, frame.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                overlay,
                f"Confidence: {event['confidence']:.2f}",
                (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            clip[i] = cv2.addWeighted(frame, 1, overlay, 0.7, 0)

        return clip

    def _combine_clips(
        self,
        clips: List[np.ndarray],
        output_path: Path
    ) -> None:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.config.fps,
            self.config.target_size
        )

        try:
            for clip in clips:
                for frame in clip:
                    if frame.shape[:2] != self.config.target_size:
                        frame = cv2.resize(frame, self.config.target_size)

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)

        finally:
            writer.release()

# Usage:
#
# python clip_generator.py baseball_game.mp4

def main():
    parser = argparse.ArgumentParser(description='Generate video highlights')

    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Path to trained model checkpoint (default: best_model.pth)')
    parser.add_argument('--output', type=str, default='highlights.mp4',
                       help='Output video name (default: highlights.mp4)')
    parser.add_argument('--min-score', type=float, default=1.0,
                       help='Minimum score threshold')
    parser.add_argument('--confidence', type=float, default=0.8,
                       help='Minimum detection confidence')

    args = parser.parse_args()

    try:
        config = ClipConfig(
            min_score=args.min_score,
            confidence_threshold=args.confidence
        )

        generator = HighlightGenerator(
            model_path=args.model,
            config=config
        )

        output_path = generator.generate_highlights(
            args.video_path,
            args.output
        )

        if output_path:
            print(f"Highlights generated successfully: {output_path}")
        else:
            print("No highlights were generated")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()