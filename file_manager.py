from pathlib import Path
from typing import Dict, List, Set

class FileManager:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.training_videos_dir = self.base_dir / "training_videos"
        self.processed_frames_dir = self.base_dir / "processed_frames"
        self.train_dir = self.processed_frames_dir / "train"
        self.val_dir = self.processed_frames_dir / "val"
        self.test_dir = self.processed_frames_dir / "test"
        self.labels_dir = self.base_dir / "labels"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.scores_dir = self.base_dir / "game_scores"
        
        for directory in [self.training_videos_dir, self.processed_frames_dir, 
                         self.train_dir, self.val_dir, self.test_dir,
                         self.labels_dir, self.checkpoints_dir, self.scores_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_video_files(self) -> List[Path]:
        return sorted(list(self.training_videos_dir.glob("*.mp4")))
    
    def get_game_ids(self) -> Set[str]:
        return {video_file.stem for video_file in self.get_video_files()}
    
    def get_data_split(self, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Dict[str, List[str]]:
        game_ids = list(self.get_game_ids())
        total_games = len(game_ids)
        
        train_size = int(total_games * train_ratio)
        val_size = int(total_games * val_ratio)
        
        train_ids = game_ids[:train_size]
        val_ids = game_ids[train_size:train_size + val_size]
        test_ids = game_ids[train_size + val_size:]
        
        return {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }
    
    def get_frame_dir(self, game_id: str) -> Path:
        return self.processed_frames_dir / game_id
    
    def get_labels_file(self, split: str) -> Path:
        return self.labels_dir / f"{split}_labels.json"
    
    def get_score_file(self, game_id: str) -> Path:
        return self.scores_dir / f"{game_id}_score.json"