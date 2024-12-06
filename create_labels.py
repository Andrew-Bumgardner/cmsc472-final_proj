#!/usr/bin/env python3

import json
import logging
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
        # 1 - home run, 2 - out, 3 - hit
        game_events = {
            "j1JMSTd71nE": [
                (20, 2), (36, 2), (51, 3), (83, 1), (92, 3), (100, 3), (111, 2), (145, 2), (162, 2), (169, 2), (179, 2), (197, 1)
            ],
            "eAdBO0peh78": [
                (62, 3), (85, 3), (106, 2), (157, 3), (188, 3), (198, 2), (204, 2), (222, 3), (241, 2), (279, 3), (300, 3), (322, 2), (337, 2)
            ],
            "T8wlAEWByv0": [
                (17, 2), (117, 3), (128, 3), (158, 3), (180, 3), (202, 1), (220, 3), (234, 1), (312, 2), (328, 2), (341, 3), (368, 2), (381, 2)
            ],
            "Dbwa-FUtF_w": [
                (27, 2), (41, 2), (62, 2), (69, 2), (95, 3), (108, 2), (122, 3), (146, 2), (154, 2), (159, 3), (162, 3), (181, 2), (202, 3), (210, 3), (231, 1), (255, 3), (262, 2), (273, 3), (280, 2), (289, 3), (302, 3), (319, 2), (322, 2), (325, 2), (336, 2), (378, 3), (386, 2), (392, 2), (408, 2), (425, 3), (460, 1), (485, 3)
            ],
            "CbiRxl4OMPc": [
                (45, 1), (66, 2), (99, 2), (116, 2), (121, 2), (130, 2), (139, 2), (163, 3), (185, 2), (188, 2), (208, 3), (238, 3), (301, 3), (342, 3), (372, 3), (387, 2), (407, 2), (433, 1)
            ],
            "Jr5AiJ0K9Og": [
                (12, 3), (22, 1), (50, 2), (69, 3), (94, 2), (101, 2), (105, 2), (117, 1), (127, 2), (148, 2), (158, 3), (170, 3), (186, 2), (197, 2), (203, 2), (228, 3), (244, 1), (257, 2), (267, 3), (270, 2), (283, 2), (299, 3), (309, 2), (314, 3), (325, 3), (348, 3), (354, 2), (369, 2), (383, 2), (389, 2), (397, 2), (408, 3), (433, 3), (449, 3), (455, 3), (460, 2), (478, 3), (484, 2), (498, 3)
            ],
            "9vZVDUjWerI": [
                (56, 3), (69, 1), (83, 3), (87, 2), (120, 2), (139, 2), (154, 3), (185, 2), (191, 2), (201, 3), (244, 1), (286, 1), (471, 1), (474, 2), (493, 1), (520, 2), (527, 2), (536, 2), (542, 2), (550, 3), (607, 1), (627, 3), (638, 3)
            ],
            "9RT_YuhXQ5I": [
                (20, 1), (35, 3), (56, 2), (87, 3), (100, 2), (105, 3), (114, 3), (137, 3), (153, 3), (166, 1), (175, 3), (212, 2), (270, 3), (299, 1), (319, 3)
            ],
            "6Kfm_u5IHE0": [
                (13, 1), (56, 1), (75, 2), (88, 2), (106, 3), (115, 2), (146, 3), (151, 2), (164, 3), (179, 3), (186, 3), (189, 3), (258, 3), (298, 2), (307, 2), (333, 1), (345, 2), (405, 3), (418, 3)
            ],
            "xeB5x246rTY": [
                (22, 3), (37, 3), (64, 2), (82, 3), (109, 1), (134, 3), (179, 3), (243, 3), (263, 3), (276, 3), (292, 3), (315, 1), (336, 2), (345, 3), (383, 1), (405, 1), (419, 3), (436, 2), (439, 2), (460, 3), (471, 3)
            ],
            "jSgk6wjhY7Q": [
                (37, 2), (42, 3), (72, 3), (92, 3), (99, 2), (121, 2), (126, 2), (134, 3), (157, 2), (200, 3), (220, 2), (232, 3), (248, 3), (260, 1), (284, 3), (290, 3), (303, 3), (331, 3), (335, 2), (347, 3), (356, 2), (359, 2), (373, 3), (439, 3), (457, 2), (470, 3), (505, 1)
            ],
            "j3ykZoQMJLI": [
                (80, 1), (101, 1), (117, 2), (120, 2), (142, 3), (155, 3), (194, 1), (229, 3), (238, 3), (264, 2), (271, 2), (284, 3), (304, 3), (319, 3), (408, 2), (416, 3), (424, 3), (493, 2), (511, 2), (513, 2)
            ],
            "Xao17c1lpPM": [
                (28, 3), (39, 3), (82, 2), (97, 2), (110, 3), (125, 2), (164, 2), (171, 3), (191, 2), (213, 3), (230, 3), (239, 3), (297, 3), (307, 2), (312, 3), (330, 3), (389, 3), (401, 3), (421, 3), (442, 2), (478, 2), (507, 2), (512, 2)
            ],
            "QUdBU7SSrdc": [
                (49, 2), (80, 2), (94, 2), (102, 3), (122, 3), (160, 2), (166, 2), (173, 2), (183, 3), (213, 2), (223, 2), (236, 3), (252, 2), (259, 2), (265, 3), (310, 3), (342, 3), (371, 2), (399, 3), (411, 3), (429, 1), (440, 2), (448, 2)
            ],
            "LlRSmfcPAZs": [
                (72, 2), (82, 2), (96, 2), (117, 1), (191, 2), (221, 3), (255, 1), (267, 2), (272, 2), (292, 2), (300, 2), (310, 2), (313, 2)
            ],
            "HqKKpT3q2Gg": [
                (23, 3), (34, 3), (54, 2), (89, 2), (99, 2), (109, 2), (113, 2), (137, 2), (150, 3), (175, 2), (183, 2), (228, 3), (251, 3), (262, 2), (269, 3), (279, 3), (294, 3), (308, 1), (316, 3), (358, 2), (393, 2), (402, 3), (449, 2), (460, 1)
            ],
            "EZXLSQDdeCk": [
                (46, 3), (73, 3), (78, 2), (85, 3), (95, 3), (137, 3), (155, 3), (159, 2), (172, 3), (187, 2), (198, 1), (221, 1), (235, 3), (241, 2), (249, 1), (288, 3), (319, 3), (330, 3), (356, 3), (359, 3), (382, 2), (387, 2), (396, 2), (405, 3), (417, 3), (436, 3), (461, 3), (469, 3)
            ],
            "D6VPcQ1Braw": [
                (88, 3), (152, 1), (182, 1), (197, 3), (209, 1), (228, 1), (245, 3), (270, 3), (355, 2), (387, 3), (409, 2), (419, 3), (441, 3), (465, 2)
            ],
            "CEEt1qOLU-w": [
                (41, 1), (101, 2), (106, 3), (116, 1), (129, 3), (133, 3), (214, 3), (224, 3), (233, 2), (277, 3), (287, 1), (312, 1), (321, 1), (333, 1), (347, 1), (349, 2)
            ],
            "9N0eR-d_LPg": [
                (17, 3), (31, 3), (48, 2), (64, 3), (93, 3), (124, 2), (131, 2), (160, 3), (192, 3), (204, 3), (212, 2), (219, 2), (235, 3), (245, 2), (284, 1), (305, 1), (331, 2), (342, 3), (349, 2), (361, 3), (393, 3), (405, 1), (421, 3), (439, 3), (442, 3), (462, 2), (468, 3), (471, 3), (494, 3), (526, 2)
            ],
            "32ii7tSxy_c": [
                (92, 3), (97, 2), (111, 3), (143, 2), (155, 3), (164, 3), (197, 3), (206, 3), (216, 3), (243, 2), (254, 1), (270, 3), (285, 3)
            ],
        }

        for split_name in ['train', 'val', 'test']:
            labels = {}
            split_dir = {
                'train': self.file_manager.train_dir,
                'val': self.file_manager.val_dir,
                'test': self.file_manager.test_dir
            }[split_name]

            if split_dir.exists():
                self.logger.info(f"Processing {split_name} directory: {split_dir}")
                game_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

                for game_dir in game_dirs:
                    game_id = game_dir.name
                    self.logger.info(f"Processing game {game_id} in {split_name} set")

                    frame_files = list(game_dir.glob('frame_*.npy'))
                    total_frames = len(frame_files)

                    if total_frames == 0:
                        self.logger.warning(f"No frames found for game {game_id}")
                        continue

                    try:
                        events = game_events.get(game_id, [])
                        if not events:
                            self.logger.warning(f"No events found for game {game_id}")
                            continue

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
            else:
                self.logger.warning(f"Directory not found for {split_name} set: {split_dir}")

    def print_dataset_stats(self, labels: Dict[str, Dict[str, int]]) -> None:
        total_stats = {event_id: 0 for event_id in self.config.event_types}

        for game_name, game_labels in labels.items():
            game_stats = {event_id: 0 for event_id in self.config.event_types}

            for event_type in game_labels.values():
                game_stats[event_type] += 1
                total_stats[event_type] += 1

            self.logger.info(f"{game_name}:")
            for event_id, event_name in self.config.event_types.items():
                self.logger.info(f"  {event_name}: {game_stats[event_id]} frames")

        self.logger.info("Total Statistics:")
        for event_id, event_name in self.config.event_types.items():
            self.logger.info(f"Total {event_name} frames: {total_stats[event_id]}")

        total_frames = sum(total_stats.values())
        event_frames = sum(count for event_id, count in total_stats.items() if event_id != 0)
        if total_frames > 0:
            self.logger.info(f"Event ratio: {event_frames / total_frames:.2%}")

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