#!/usr/bin/env python3

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from file_manager import FileManager

@dataclass
class ScoringConfig:
    event_types: List[str]

    event_scores: Dict[str, float]
    confidence_thresholds: Dict[str, float]
    temporal_weights: Dict[str, float]

    multipliers: Dict[str, Dict[str, float]]

    analysis_window: int = 100
    min_events_for_stats: int = 10

    def __post_init__(self):
        for event_type in self.event_types:
            if event_type not in self.event_scores:
                raise ValueError(f"Missing score for event type: {event_type}")

        for event_type in self.event_scores:
            if event_type not in self.event_types:
                raise ValueError(f"Score defined for invalid event type: {event_type}")

        for event_type in self.multipliers:
            if event_type not in self.event_types:
                raise ValueError(f"Multipliers defined for invalid event type: {event_type}")

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ScoringConfig':
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            required_fields = [
                'event_types',
                'event_scores',
                'confidence_thresholds', 
                'temporal_weights',
                'multipliers'
            ]

            for field in required_fields:
                if field not in config_data:
                    raise ValueError(f"Missing required field in config: {field}")

            return cls(**config_data)

        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {str(e)}")

            return cls(
                event_types=['no_event', 'home_run', 'out', 'hit'],
                event_scores={"home_run": 4.0, "out": 2.0, 'hit': 1.0, "no_event": 0.0},
                confidence_thresholds={"min": 0.7, "high": 0.9},
                temporal_weights={"start": 1.0, "middle": 1.1, "end": 1.2},
                multipliers={
                    "home_run": {"confidence": 1.2, "late_game": 1.3, "sequence": 1.1},
                    "out": {"confidence": 1.1, "late_game": 1.2, "sequence": 1.2},
                    "hit": {"confidence": 1.1, "late_game": 1.2, "sequence": 1.2}
                }
            )

class EventScorer:
    def __init__(
        self,
        config_path: str,
        scores_dir: str = "game_scores"
    ):
        self.config = ScoringConfig.from_yaml(config_path)
        self.file_manager = FileManager()

        self.scores_dir = Path(scores_dir)
        self.scores_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.metrics = {
            'processed_games': 0,
            'total_events': 0,
            'event_distribution': {},
            'score_history': []
        }

        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })

    def calculate_event_score(
        self,
        event: Dict,
        game_context: Optional[Dict] = None,
        previous_events: Optional[List[Dict]] = None
    ) -> float:
        try:
            event_type = event.get('event_type', 'no_event')
            base_score = self.config.event_scores.get(event_type, 0.0)

            if base_score == 0:
                return 0.0

            multipliers = self.config.multipliers.get(event_type, {})
            final_score = base_score

            confidence = event.get('confidence', 0.0)
            if confidence < self.config.confidence_thresholds['min']:
                return 0.0

            if confidence >= self.config.confidence_thresholds['high']:
                final_score *= multipliers.get('confidence', 1.0)

            if game_context:
                inning = game_context.get('inning', 1)
                if inning >= 7:
                    final_score *= multipliers.get('late_game', 1.0)

            if previous_events:
                consecutive_count = 0
                for prev_event in reversed(previous_events):
                    if event['frame_idx'] - prev_event['frame_idx'] > 300:
                        break

                    if prev_event['event_type'] == event_type:
                        consecutive_count += 1
                    else:
                        break

                if consecutive_count > 0:
                    sequence_multiplier = min(1.0 + (consecutive_count * 0.1), 1.3)
                    final_score *= sequence_multiplier

            if game_context and 'total_frames' in game_context:
                frame_idx = event['frame_idx']
                total_frames = game_context['total_frames']
                phase = self._determine_game_phase(frame_idx, total_frames)
                final_score *= self.config.temporal_weights[phase]

            return round(final_score, 2)

        except Exception as e:
            self.logger.error(f"Error calculating event score: {str(e)}")
            return 0.0

    def _determine_game_phase(self, frame_idx: int, total_frames: int) -> str:
        progress = frame_idx / total_frames
        if progress < 0.33:
            return "start"
        elif progress < 0.66:
            return "middle"
        return "end"

    def score_game(
        self,
        game_id: str,
        predictions_file: str,
        game_context_file: Optional[str] = None
    ) -> Dict:
        try:
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)

            game_context = None
            if game_context_file and Path(game_context_file).exists():
                with open(game_context_file, 'r') as f:
                    game_context = json.load(f)

            scored_events = []
            event_counts = {event_type: 0 for event_type in self.config.event_scores.keys()}
            total_score = 0.0

            for i, event in enumerate(predictions):
                previous_events = predictions[:i] if i > 0 else None

                event_score = self.calculate_event_score(
                    event, game_context, previous_events
                )

                event_type = event.get('event_type', 'no_event')
                event_counts[event_type] += 1
                total_score += event_score

                scored_event = {
                    **event,
                    'score': event_score,
                    'timestamp': self._frame_to_time(event['frame_idx']),
                    'cumulative_score': round(total_score, 2)
                }
                scored_events.append(scored_event)

            analytics = self._generate_game_analytics(scored_events)

            summary = {
                'game_id': game_id,
                'total_score': round(total_score, 2),
                'event_counts': event_counts,
                'scored_events': scored_events,
                'analytics': analytics,
                'timestamp': datetime.now().isoformat()
            }

            self._update_metrics(summary)

            self._save_game_score(game_id, summary)
            self._generate_game_visualizations(game_id, summary)

            return summary

        except Exception as e:
            self.logger.error(f"Error scoring game {game_id}: {str(e)}")
            raise

    def _generate_game_analytics(
        self,
        scored_events: List[Dict]
    ) -> Dict:
        if not scored_events:
            return {}

        scores = [event['score'] for event in scored_events]

        total_frames = max(event['frame_idx'] for event in scored_events) + 1 if scored_events else None

        return {
            'score_statistics': {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'max': np.max(scores),
                'min': np.min(scores)
            },
            'event_density': {
                'early_game': len([e for e in scored_events 
                    if self._frame_to_time_ratio(e['frame_idx'], total_frames) < 0.33]),
                'mid_game': len([e for e in scored_events 
                    if 0.33 <= self._frame_to_time_ratio(e['frame_idx'], total_frames) < 0.66]),
                'late_game': len([e for e in scored_events 
                    if self._frame_to_time_ratio(e['frame_idx'], total_frames) >= 0.66])
            },
            'confidence_metrics': {
                'mean': np.mean([e.get('confidence', 0) for e in scored_events]),
                'high_confidence_ratio': len([e for e in scored_events 
                    if e.get('confidence', 0) >= self.config.confidence_thresholds['high']]) / len(scored_events)
            }
        }

    def process_all_games(self) -> Dict:
        predictions_dir = self.file_manager.predictions_dir
        game_files = list(predictions_dir.glob('**/*_predictions.json'))

        if not game_files:
            self.logger.warning("No prediction files found")
            return {}

        results = {}
        for pred_file in game_files:
            game_id = pred_file.stem.replace('_predictions', '')
            context_file = pred_file.parent / f"{game_id}_context.json"

            try:
                results[game_id] = self.score_game(
                    game_id,
                    str(pred_file),
                    str(context_file) if context_file.exists() else None
                )
                self.logger.info(f"Processed game: {game_id}")

            except Exception as e:
                self.logger.error(f"Failed to process game {game_id}: {str(e)}")
                continue

        summary = self.generate_summary_report(results)
        self._save_summary_report(summary)

        return results

    def generate_summary_report(self, results: Dict[str, Dict]) -> Dict:
        if not results:
            return {"error": "No games processed"}

        total_events = sum(game['event_counts']['home_run'] + game['event_counts']['out'] + game['event_counts']['hit']
                          for game in results.values())
        total_score = sum(game['total_score'] for game in results.values())
        num_games = len(results)

        summary = {
            "overall_statistics": {
                "total_games": num_games,
                "total_events": total_events,
                "total_score": round(total_score, 2),
                "average_score_per_game": round(total_score / num_games, 2),
                "average_events_per_game": round(total_events / num_games, 2)
            },
            "event_statistics": self._calculate_event_statistics(results),
            "confidence_analysis": self._analyze_confidence_distribution(results),
            "temporal_analysis": self._analyze_temporal_distribution(results),
            "generated_at": datetime.now().isoformat()
        }

        self._generate_summary_visualizations(summary, results)

        return summary

    def _calculate_event_statistics(self, results: Dict[str, Dict]) -> Dict:
        event_totals = {}
        event_scores = {}

        for game in results.values():
            for event_type, count in game['event_counts'].items():
                event_totals[event_type] = event_totals.get(event_type, 0) + count

                for event in game['scored_events']:
                    if event['event_type'] == event_type:
                        if event_type not in event_scores:
                            event_scores[event_type] = []
                        event_scores[event_type].append(event['score'])

        event_stats = {}
        for event_type, scores in event_scores.items():
            if scores:
                event_stats[event_type] = {
                    'total_count': event_totals[event_type],
                    'average_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'max_score': np.max(scores)
                }

        return event_stats

    def _update_metrics(self, game_summary: Dict):
        self.metrics['processed_games'] += 1
        self.metrics['total_events'] += sum(game_summary['event_counts'].values())

        for event_type, count in game_summary['event_counts'].items():
            self.metrics['event_distribution'][event_type] = \
                self.metrics['event_distribution'].get(event_type, 0) + count

        self.metrics['score_history'].append(game_summary['total_score'])

    def _frame_to_time(self, frame_idx: int, fps: int = 30) -> str:
        total_seconds = frame_idx // fps
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _frame_to_time_ratio(self, frame_idx: int, total_frames: Optional[int] = None) -> float:
        try:
            if total_frames is None or total_frames <= 0:
                total_frames = 30 * 60 * 10

            frame_idx = max(0, min(frame_idx, total_frames))

            return frame_idx / total_frames

        except Exception as e:
            self.logger.warning(f"Error calculating time ratio: {str(e)}, using 0.0")
            return 0.0

    def _save_game_score(self, game_id: str, summary: Dict):
        output_file = self.scores_dir / f"{game_id}_score.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def _save_summary_report(self, summary: Dict):
        output_file = self.scores_dir / "summary_report.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def _frame_to_time_str(self, frame_idx: int, fps: int = 30) -> str:
        total_seconds = frame_idx // fps
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _generate_game_visualizations(self, game_id: str, summary: Dict):
        plt.figure()
        events = summary['event_counts']
        plt.bar(events.keys(), events.values(), color=['#2ecc71', '#e74c3c', '#3498db', '#f1c40f'])
        plt.title(f'Event Distribution - Game {game_id}')
        plt.xlabel('Event Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.scores_dir / f"{game_id}_events.png")
        plt.close()

        if summary['scored_events']:
            plt.figure(figsize=(12, 6))
            timestamps = [event['frame_idx'] / 30 for event in summary['scored_events']]  # assuming 30 fps
            scores = [event['cumulative_score'] for event in summary['scored_events']]
            event_types = [event['event_type'] for event in summary['scored_events']]

            plt.plot(timestamps, scores, 'b-', label='Cumulative Score', linewidth=2)

            colors = {'home_run': 'red', 'out': 'green', 'hit': 'blue'}
            for i, (t, score, event_type) in enumerate(zip(timestamps, scores, event_types)):
                plt.scatter(t, score, c=colors.get(event_type, 'gray'), marker='o', s=100)

            plt.title(f'Score Timeline - Game {game_id}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Cumulative Score')

            max_time = max(timestamps)
            tick_positions = np.linspace(0, max_time, 10)
            tick_labels = [self._frame_to_time_str(int(t * 30)) for t in tick_positions]
            plt.xticks(tick_positions, tick_labels, rotation=45)

            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.scores_dir / f"{game_id}_timeline.png")
            plt.close()

        if summary['scored_events']:
            plt.figure()
            confidences = [event.get('confidence', 0) for event in summary['scored_events']]
            scores = [event['score'] for event in summary['scored_events']]

            plt.hist2d(confidences, scores, bins=20, cmap='YlOrRd')
            plt.colorbar(label='Count')
            plt.title(f'Confidence vs Score Distribution - Game {game_id}')
            plt.xlabel('Confidence')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(self.scores_dir / f"{game_id}_confidence_dist.png")
            plt.close()

    def _generate_summary_visualizations(self, summary: Dict, results: Dict[str, Dict]):
        plots_dir = self.scores_dir / 'summary_plots'
        plots_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(12, 6))
        total_events = summary['event_statistics']
        event_names = list(total_events.keys())
        event_counts = [stats['total_count'] for stats in total_events.values()]

        bars = plt.bar(event_names, event_counts)
        plt.title('Overall Event Distribution')
        plt.xlabel('Event Type')
        plt.ylabel('Total Count')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(plots_dir / 'overall_distribution.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        game_scores = [game['total_score'] for game in results.values()]
        plt.hist(game_scores, bins=20, edgecolor='black')
        plt.title('Game Score Distribution')
        plt.xlabel('Total Game Score')
        plt.ylabel('Number of Games')
        plt.axvline(np.mean(game_scores), color='r', linestyle='dashed', 
                label=f'Mean: {np.mean(game_scores):.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'score_distribution.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        event_scores = {event_type: [] for event_type in self.config.event_scores.keys()}

        for game in results.values():
            for event in game['scored_events']:
                event_scores[event['event_type']].append(event['score'])

        valid_scores = [scores for scores in event_scores.values() if scores]
        valid_labels = [event_type for event_type, scores in event_scores.items() if scores]

        plt.boxplot(valid_scores, tick_labels=valid_labels)
        plt.title('Event Score Distribution by Type')
        plt.xlabel('Event Type')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'event_score_distribution.png')
        plt.close()

    def _analyze_confidence_distribution(self, results: Dict[str, Dict]) -> Dict:
        all_confidences = []
        confidence_by_type = {}

        for game in results.values():
            for event in game['scored_events']:
                conf = event.get('confidence', 0)
                event_type = event['event_type']

                all_confidences.append(conf)
                if event_type not in confidence_by_type:
                    confidence_by_type[event_type] = []
                confidence_by_type[event_type].append(conf)

        analysis = {
            'overall': {
                'mean': np.mean(all_confidences),
                'median': np.median(all_confidences),
                'std': np.std(all_confidences),
                'high_confidence_ratio': len([c for c in all_confidences 
                    if c >= self.config.confidence_thresholds['high']]) / len(all_confidences)
            },
            'by_type': {}
        }

        for event_type, confidences in confidence_by_type.items():
            analysis['by_type'][event_type] = {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences)
            }

        return analysis

    def _analyze_temporal_distribution(self, results: Dict[str, Dict]) -> Dict:
        early_game_events = {event_type: 0 for event_type in self.config.event_types}
        mid_game_events = {event_type: 0 for event_type in self.config.event_types}
        late_game_events = {event_type: 0 for event_type in self.config.event_types}

        for game in results.values():
            total_frames = (max(event['frame_idx'] for event in game['scored_events']) + 1 
                        if game['scored_events'] else None)

            for event in game['scored_events']:
                progress = self._frame_to_time_ratio(event['frame_idx'], total_frames)
                event_type = event['event_type']

                if progress < 0.33:
                    early_game_events[event_type] += 1
                elif progress < 0.66:
                    mid_game_events[event_type] += 1
                else:
                    late_game_events[event_type] += 1

        return {
            'early_game': early_game_events,
            'mid_game': mid_game_events,
            'late_game': late_game_events
        }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Score baseball game events')
    parser.add_argument('--config', type=str, default='scoring_config.yaml',
                       help='Path to scoring configuration file')
    parser.add_argument('--output-dir', type=str, default='game_scores',
                       help='Directory for output files')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('scoring_detailed.log'),
            logging.StreamHandler()
        ]
    )

    try:
        scorer = EventScorer(args.config, args.output_dir)

        results = scorer.process_all_games()
        if results:
            print("\nProcessing Summary:")
            print(f"Total games processed: {len(results)}")
            print("\nEvent counts across all games:")
            total_events = {}
            for game in results.values():
                for event_type, count in game['event_counts'].items():
                    if event_type != 'no_event':
                        total_events[event_type] = total_events.get(event_type, 0) + count

            for event_type, count in total_events.items():
                print(f"{event_type}: {count}")
        else:
            print("No games were processed.")

    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()