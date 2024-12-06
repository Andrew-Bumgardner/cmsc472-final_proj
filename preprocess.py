#!/usr/bin/env python3

import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from file_manager import FileManager

@dataclass
class PreprocessingConfig:
    target_size: Tuple[int, int] = (640, 480)
    sample_rate: int = 30  # Process every nth frame
    max_frames: int = 10000
    quality_threshold: float = 0.4
    batch_size: int = 16
    num_workers: int = 4
    cache_size: int = 500
    min_brightness: int = 30
    max_brightness: int = 225
    blur_threshold: float = 50.0

    def validate(self):
        assert all(x > 0 for x in self.target_size), "Invalid target size"
        assert 1 <= self.sample_rate <= 60, "Sample rate must be between 1 and 60"
        assert 100 <= self.max_frames <= 15000, "Max frames must be between 100 and 15000"
        assert 0 <= self.quality_threshold <= 1, "Invalid quality threshold"
        assert 1 <= self.batch_size <= 32, "Batch size must be between 1 and 32"
        assert 1 <= self.num_workers <= 8, "Number of workers must be between 1 and 8"
        assert self.cache_size >= 0, "Invalid cache size"

class FrameCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = {}
        self.usage_order = []

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            self.usage_order.remove(key)
            self.usage_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, frame: np.ndarray):
        if self.max_size <= 0:
            return

        if len(self.cache) >= self.max_size:
            lru_key = self.usage_order.pop(0)
            self.cache.pop(lru_key)

        self.cache[key] = frame
        self.usage_order.append(key)

    def clear(self):
        self.cache.clear()
        self.usage_order.clear()

class VideoPreprocessor:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config: Optional[PreprocessingConfig] = None
    ):
        self.file_manager = FileManager()
        self.split = self.file_manager.get_data_split()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or PreprocessingConfig()
        self.config.validate()

        self.frame_cache = FrameCache(self.config.cache_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = self._setup_transforms()

        self._setup_logging()

        self.metrics = {
            'processed_videos': 0,
            'processed_frames': 0,
            'failed_frames': 0,
            'processing_time': 0,
            'skipped_frames': 0,
            'cached_frames': 0,
            'low_quality_frames': 0
        }

    def _setup_transforms(self) -> transforms.Compose:
        try:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.config.target_size),
                transforms.ToTensor(),
            ])
        except Exception as e:
            self.logger.error(f"Error setting up transforms: {str(e)}")
            raise

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/preprocessing.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _assess_frame_quality(self, frame: np.ndarray) -> Tuple[float, str]:
        try:
            if frame is None or frame.size == 0:
                return 0.0, "Empty frame"

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            brightness = np.mean(gray)
            if brightness < self.config.min_brightness:
                return 0.0, "Too dark"
            if brightness > self.config.max_brightness:
                return 0.0, "Too bright"

            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < self.config.blur_threshold:
                return 0.0, "Too blurry"

            quality_score = min(1.0, blur_score / 1000)

            return quality_score, "OK"

        except Exception as e:
            self.logger.warning(f"Frame quality assessment failed: {str(e)}")
            return 0.0, f"Error: {str(e)}"

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        output_dir: Path
    ) -> bool:
        try:
            cache_key = f"{output_dir}_{frame_index}"
            cached_frame = self.frame_cache.get(cache_key)
            if cached_frame is not None:
                self.metrics['cached_frames'] += 1
                return True

            quality_score, quality_status = self._assess_frame_quality(frame)
            if quality_score < self.config.quality_threshold:
                self.metrics['low_quality_frames'] += 1
                self.logger.debug(f"Frame {frame_index} failed quality check: {quality_status}")
                return False

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.transform(rgb_frame).numpy()

            output_path = output_dir / f"frame_{frame_index:06d}.npy"
            np.save(str(output_path), processed_frame)
            output_path = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), frame)

            self.frame_cache.put(cache_key, processed_frame)

            return True

        except Exception as e:
            self.logger.warning(f"Frame processing failed: {str(e)}")
            self.metrics['failed_frames'] += 1
            return False

    def preprocess_video(self, video_path: Path, output_dir: Path) -> bool:
        try:
            if not self._validate_video(video_path):
                return False

            start_time = time.time()

            output_dir.mkdir(exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            processed_count = 0
            frame_count = 0

            actual_frames = (total_frames + self.config.sample_rate - 1) // self.config.sample_rate
            with tqdm(total=actual_frames, desc=f"{video_path.stem}", unit=" frames") as pbar:
                while cap.isOpened() and processed_count < self.config.max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % self.config.sample_rate == 0:
                        if self._process_frame(frame, processed_count, output_dir):
                            processed_count += 1
                            pbar.update(1)
                    else:
                        self.metrics['skipped_frames'] += 1

                    frame_count += 1

            processing_time = time.time() - start_time
            self.metrics['processed_videos'] += 1
            self.metrics['processed_frames'] += processed_count
            self.metrics['processing_time'] += processing_time

            return True

        except Exception as e:
            self.logger.error(f"Error processing {video_path}: {str(e)}")
            return False

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            self.frame_cache.clear()

    def _validate_video(self, video_path: Path) -> bool:
        try:
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            if video_path.stat().st_size < 1024:
                raise ValueError(f"Video file too small: {video_path}")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Unable to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not (10 <= fps <= 60):
                self.logger.warning(f"Unusual FPS: {fps} for {video_path}")

            if frame_count < 30:
                raise ValueError(f"Video too short: {frame_count} frames")

            if frame_width < 100 or frame_height < 100:
                raise ValueError(f"Video resolution too low: {frame_width}x{frame_height}")

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                raise ValueError("Unable to read first frame")

            return True

        except Exception as e:
            self.logger.error(f"Video validation failed: {str(e)}")
            return False

        finally:
            if 'cap' in locals():
                cap.release()

    def process_all_videos(self) -> Dict:
        video_files = self.file_manager.get_video_files()

        if not video_files:
            raise ValueError(f"No MP4 files found in {self.file_manager.training_videos_dir}")

        self.logger.info(f"Found {len(video_files)} videos to process")

        processing_args = []
        for video_file in video_files:
            game_id = video_file.stem
            if game_id in self.split['train']:
                output_dir = self.file_manager.train_dir / game_id
            elif game_id in self.split['val']:
                output_dir = self.file_manager.val_dir / game_id
            else:
                output_dir = self.file_manager.test_dir / game_id

            processing_args.append((video_file, output_dir))

        start_time = time.time()
        failed_videos = []

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for args in processing_args:
                future = executor.submit(self.preprocess_video, *args)
                futures.append((args[0], future))

            for video_path, future in tqdm(futures, desc="Processing videos"):
                try:
                    if not future.result():
                        failed_videos.append(video_path)
                except Exception as e:
                    self.logger.error(f"Error processing {video_path}: {str(e)}")
                    failed_videos.append(video_path)

        total_time = time.time() - start_time
        report = {
            'total_videos': self.metrics['processed_videos'],
            'total_frames': self.metrics['processed_frames'],
            'failed_frames': self.metrics['failed_frames'],
            'low_quality_frames': self.metrics['low_quality_frames'],
            'skipped_frames': self.metrics['skipped_frames'],
            'cached_frames': self.metrics['cached_frames'],
            'average_fps': self.metrics['processed_frames'] / total_time,
            'total_time': total_time,
            'failed_videos': [str(path) for path in failed_videos],
            'failure_rate': len(failed_videos) / len(video_files)
        }

        self._log_processing_summary(report)

        return report

    def _log_processing_summary(self, report: Dict):
        self.logger.info("\nProcessing Summary:")
        self.logger.info(f"Total videos processed: {report['total_videos']}")
        self.logger.info(f"Total frames processed: {report['total_frames']}")
        self.logger.info(f"Failed frames: {report['failed_frames']}")
        self.logger.info(f"Low quality frames: {report['low_quality_frames']}")
        self.logger.info(f"Skipped frames: {report['skipped_frames']}")
        self.logger.info(f"Cached frames: {report['cached_frames']}")
        self.logger.info(f"Average FPS: {report['average_fps']:.2f}")
        self.logger.info(f"Total processing time: {report['total_time']:.2f} seconds")

        if report['failed_videos']:
            self.logger.warning("\nFailed videos:")
            for video_path in report['failed_videos']:
                self.logger.warning(f"- {video_path}")
            self.logger.warning(f"Failure rate: {report['failure_rate']:.2%}")

    def cleanup(self):
        try:
            self.frame_cache.clear()

            report_path = self.output_dir / "processing_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)

            self.logger.info(f"Processing report saved to {report_path}")

        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Preprocess baseball game videos')
    parser.add_argument('--input-dir', type=str, default='training_videos',
                       help='Input directory containing training videos')
    parser.add_argument('--output-dir', type=str, default='processed_frames',
                       help='Output directory for processed frames')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--max-frames', type=int, help='Override maximum frames to process')
    parser.add_argument('--quality-threshold', type=float, 
                       help='Override quality threshold (0.0-1.0)')
    parser.add_argument('--sample-rate', type=int,
                       help='Override frame sampling rate')

    args = parser.parse_args()

    try:
        config = PreprocessingConfig()
        if args.config:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

        if args.max_frames:
            config.max_frames = args.max_frames
        if args.quality_threshold:
            config.quality_threshold = args.quality_threshold
        if args.sample_rate:
            config.sample_rate = args.sample_rate

        preprocessor = VideoPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=config
        )

        try:
            report = preprocessor.process_all_videos()

            report_path = Path(args.output_dir) / "processing_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\nProcessing report saved to {report_path}")

            print("\nProcessing Summary:")
            print(f"Total videos processed: {report['total_videos']}")
            print(f"Total frames processed: {report['total_frames']}")
            print(f"Average FPS: {report['average_fps']:.2f}")

            if report['failed_videos']:
                print(f"\nWarning: {len(report['failed_videos'])} videos failed to process")
                print(f"Failure rate: {report['failure_rate']:.2%}")

        finally:
            preprocessor.cleanup()

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)

    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()