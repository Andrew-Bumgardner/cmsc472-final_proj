#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from file_manager import FileManager

@dataclass
class LabelingConfig:
    sequence_length: int = 5
    event_types: Dict[int, str] = None
    
    def __post_init__(self):
        if self.event_types is None:
            self.event_types = {
                0: 'no_event',
                1: 'home_run',
                2: 'out',
                3: 'hit'
            }

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")

class LabelCreator:
    def __init__(self, config: Optional[LabelingConfig] = None):
        self.file_manager = FileManager()
        self.config = config or LabelingConfig()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_event_sequence(
        self,
        start_frame: int,
        event_type: int,
        total_frames: int
    ) -> Dict[str, int]:
        if event_type not in self.config.event_types:
            raise ValueError(f"Invalid event type: {event_type}")
            
        sequence = {}
        end_frame = min(start_frame + self.config.sequence_length, total_frames)
        
        for i in range(start_frame, end_frame):
            sequence[str(i)] = event_type
            
        return sequence

    def create_game_labels(self, game_events: List[Tuple[int, int]], total_frames: int) -> Dict[str, int]:
        labels = {str(i): 0 for i in range(total_frames)}
        
        game_events = sorted(game_events, key=lambda x: x[0])
        
        for i in range(len(game_events) - 1):
            curr_frame, curr_type = game_events[i]
            next_frame, next_type = game_events[i + 1]
            
            if curr_frame + self.config.sequence_length > next_frame:
                self.logger.warning(
                    f"Overlapping events at frames {curr_frame} ({self.config.event_types[curr_type]}) "
                    f"and {next_frame} ({self.config.event_types[next_type]})"
                )
                
            frame_gap = next_frame - curr_frame
            if curr_type == next_type and frame_gap < 30:
                self.logger.warning(
                    f"Suspicious sequence: Two {self.config.event_types[curr_type]} events "
                    f"only {frame_gap} frames apart"
                )
        
        for frame_num, event_type in game_events:
            if 0 <= frame_num < total_frames:
                event_sequence = self.create_event_sequence(frame_num, event_type, total_frames)
                labels.update(event_sequence)
        
        return labels

    def get_game_frame_count(self, game_id: str) -> int:
        frame_dir = self.file_manager.train_dir / game_id
        if not frame_dir.exists():
            frame_dir = self.file_manager.val_dir / game_id
            if not frame_dir.exists():
                frame_dir = self.file_manager.test_dir / game_id
        
        if frame_dir.exists():
            frames = list(frame_dir.glob('*.npy'))
            return len(frames)
            
        self.logger.warning(
            f"Frame directory not found for game {game_id}. "
            "Using default frame count."
        )
        return 600

    def create_dataset_labels(self) -> None:
        split = self.file_manager.get_data_split()
        
        # 1 - home run, 2 - out, 3 - hit
        game_events = {
            "9vZVDUjWerI": [
                (126, 2), (127, 2), (128, 2), (139, 2), (140, 2), (141, 2), (142, 2), (162, 3), (163, 3), (164, 3), (193, 1), (194, 1), (201, 1), (202, 1), (203, 1), (204, 1)
            ],
            "CbiRxl4OMPc": [
                (111, 2), (112, 2), (113, 2), (183, 2), (184, 2), (185, 2), (186, 2), (190, 2), (191, 2), (192, 2)
            ],
            "EZXLSQDdeCk": [
                (104, 2), (128, 3), (215, 3), (232, 2), (242, 3), (279, 3), (407, 3), (457, 3), (478, 2), (511, 3), (557, 2), (576, 1), (651, 1), (701, 3), (724, 2), (736, 1), (863, 3), (987, 3), (1070, 3), (1081, 3), (1142, 2), (1168, 2), (1194, 2), (1215, 3), (1245, 3), (1305, 3), (1377, 3), (1403, 3)
            ],
            "j3ykZoQMJLI": [
                (214, 1), (274, 1), (339, 2), (349, 2), (409, 3), (447, 3), (555, 1), (671, 3), (693, 3), (734, 3), (781, 2), (802, 2), (842, 3), (895, 3), (936, 3), (1226, 2), (1242, 3), (1264, 3), (1477, 2), (1528, 2), (1536, 2)
            ],
            "Jr5AiJ0K9Og": [
                (33, 3), (51, 1), (198, 3), (333, 1), (463, 3), (580, 2), (600, 2), (698, 2), (699, 1), (755, 2), (779, 3), (790, 2), (830, 2), (867, 3), (904, 2), (913, 3), (941, 3), (997, 3), (1031, 2), (1106, 2), (1124, 2), (1149, 2), (1180, 3), (1280, 2), (1302, 3), (1315, 3), (1336, 2), (1390, 3), (1415, 2), (1453, 3)
            ],
            "xeB5x246rTY": [
                (55, 3), (96, 3), (198, 2), (246, 3), (315, 1), (399, 3), (532, 3), (732, 3), (788, 3), (825, 3), (867, 3), (932, 1), (1010, 2), (1036, 3), (1137, 1), (1202, 1), (1255, 3), (1314, 2), (1322, 2), (1361, 2), (1376, 3), (1418, 3)
            ],
            "HqKKpT3q2Gg": [
                (66, 3), (98, 3), (171, 2), (310, 2), (338, 2), (351, 2), (406, 3), (419, 2), (453, 3), (528, 2), (553, 2), (683, 3), (747, 3), (785, 2), (805, 3), (823, 3), (866, 3), (872, 3), (906, 1), (937, 3), (1066, 2), (1169, 2), (1187, 3), (1319, 2), (1353, 1)
            ],
            "D6VPcQ1Braw": [
                (145, 2), (242, 3), (425, 1), (517, 1), (579, 3), (604, 1), (663, 1), (720, 3), (787, 3), (963, 2), (1135, 3), (1210, 2), (1234, 3), (1300, 3), (1373, 2)
            ],
            "6Kfm_u5IHE0": [
                (26, 1), (144, 1), (221, 2), (231, 2), (259, 2), (310, 3), (344, 2), (431, 3), (450, 2), (476, 3), (522, 3), (544, 3), (554, 3), (635, 3), (756, 3), (883, 2), (912, 2), (968, 1), (1022, 2), (1197, 3), (1231, 3)
            ],
            "jSgk6wjhY7Q": [
                (107, 2), (119, 3), (211, 3), (270, 3), (300, 2), (365, 2), (378, 2), (402, 3), (458, 2), (471, 2), (567, 2), (580, 2), (590, 3), (655, 2), (683, 3), (739, 3), (755, 2), (768, 1), (843, 3), (864, 3), (888, 2), (903, 3), (993, 3), (1007, 2), (1038, 3), (1070, 2), (1079, 2), (1115, 3), (1190, 3), (1294, 2), (1307, 3), (1363, 2), (1397, 3), (1484, 1)
            ],
        }

        for split_name in ['train', 'val']:
            labels = {}
            split_dir = (self.file_manager.train_dir if split_name == 'train' 
                        else self.file_manager.val_dir)

            game_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            
            for game_dir in game_dirs:
                game_id = game_dir.name

                frame_files = list(game_dir.glob('frame_*.npy'))
                total_frames = len(frame_files)
                
                if total_frames == 0:
                    self.logger.warning(f"No frames found for game {game_id}")
                    continue
                
                try:
                    events = game_events.get(game_id, [])

                    game_labels = self.create_game_labels(events, total_frames)
                    labels[game_id] = game_labels
                    
                    self.logger.info(f"Created labels for game {game_id} with {total_frames} frames")
                    
                except Exception as e:
                    self.logger.error(f"Error creating labels for {game_id}: {str(e)}")
                    continue

            if labels:
                labels_file = self.file_manager.get_labels_file(split_name)
                with open(labels_file, 'w') as f:
                    json.dump(labels, f, indent=2)
                
                self.logger.info(f"Created labels for {split_name} set: {labels_file}")

                print(f"\n{split_name.capitalize()} Set Statistics:")
                self.print_dataset_stats(labels)
            else:
                self.logger.warning(f"No labels created for {split_name} set")

    def print_dataset_stats(self, labels: Dict[str, Dict[str, int]]) -> None:
        total_stats = {event_id: 0 for event_id in self.config.event_types}
        
        for game_name, game_labels in labels.items():
            game_stats = {event_id: 0 for event_id in self.config.event_types}
            
            for event_type in game_labels.values():
                game_stats[event_type] += 1
                total_stats[event_type] += 1
            
            print(f"\n{game_name}:")
            for event_id, event_name in self.config.event_types.items():
                print(f"  {event_name}: {game_stats[event_id]} frames")
        
        print("\nTotal Statistics:")
        for event_id, event_name in self.config.event_types.items():
            print(f"Total {event_name} frames: {total_stats[event_id]}")
        
        total_frames = sum(total_stats.values())
        event_frames = sum(count for event_id, count in total_stats.items() if event_id != 0)
        if total_frames > 0:
            print(f"Event ratio: {event_frames / total_frames:.2%}")

def main():
    try:
        config = LabelingConfig()
        creator = LabelCreator(config)
        creator.create_dataset_labels()
        
    except Exception as e:
        logging.error(f"Error creating labels: {str(e)}")
        raise

if __name__ == "__main__":
    main()