# Baseball Game Analysis Pipeline

A deep learning pipeline for detecting and analyzing key events in baseball game videos using video footage.

## Overview
This project uses a 3D CNN model to detect significant events (home runs, outs, and hits) in baseball game videos and provides a comprehensive scoring system for game analysis.

## Setup
```bash
brew install python@3.11
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Directory Structure
```
baseball-analysis/
├── training_videos/    # Raw input video files
├── processed_frames/   # Preprocessed video frames
│   ├── train/         # Training set frames
│   ├── val/           # Validation set frames
│   └── test/          # Test set frames
├── game_scores/       # Game scoring outputs
│   └── summary_plots/ # Performance visualizations
├── training_plots/    # Training metrics and visualizations
├── predictions/       # Model prediction outputs
└── logs/             # Execution logs
```

## Pipeline Components

### 1. Video Preprocessing (`preprocess.py`)
Converts raw game footage into processed frames suitable for analysis.

**Features:**
- Frame quality assessment
- Brightness and blur detection
- Frame caching for performance
- Multi-threaded processing

**Parameters:**
- Target size: (640, 480)
- Sample rate: 10 frames
- Maximum frames: 10,000 per video
- Quality threshold: 0.4
- Minimum brightness: 30
- Maximum brightness: 225
- Blur threshold: 50.0

### 2. Manual Annotation (`annotate.py`)
GUI tool for manually annotating game events in processed frames.

**Features:**
- Frame-by-frame navigation
- Event type selection
- Keyboard shortcuts
- Real-time annotation display
- Annotation saving/loading

### 3. Label Creation (`create_labels.py`)
Generates labeled data for model training from annotations.

**Event Types:**
- No Event (0)
- Home Run (1)
- Out (2)
- Hit (3)

**Features:**
- Sequence-based labeling
- Automatic train/val/test split
- Event overlap detection
- Suspicious sequence detection

### 4. Model Training (`train.py`)
Trains a 3D CNN model using processed frames and labels.

**Model Architecture:**
- Backbone: R3D-18
- Custom motion detection branch
- Adaptive classifier head

**Training Features:**
- Focal Loss with class weighting
- OneCycleLR scheduling (GPU)
- ReduceLROnPlateau (CPU)
- Early stopping
- Real-time training visualization

**Parameters:**
- Learning rate: 0.001
- Batch size: 4 (GPU) / 2 (CPU)
- Sequence length: 6 (GPU) / 4 (CPU)
- Number of epochs: 10
- Early stopping patience: 30

### 5. Generate Predictions (`generate_predictions.py`)
Generates event predictions for all games using the trained model.

**Features:**
- Batch processing
- Frame sequence analysis
- Confidence scoring
- JSON output format
- GPU acceleration

### 6. Game Scoring (`scoring.py`)
Analyzes and scores detected events from predictions.

**Scoring Components:**
- Base event scores
- Confidence multipliers
- Temporal weights
- Sequential event bonuses

**Analysis Features:**
- Event distribution analysis
- Temporal pattern detection
- Confidence analysis
- Comprehensive visualizations

### 7. Clip Generation (`clip_generator.py`)
Creates highlight clips based on detected events.

**Features:**
- Batch processing support
- Confidence thresholding
- Event merging logic
- Fade transitions
- Custom overlay generation

**Parameters:**
- Before event: 60 frames
- After event: 90 frames
- Minimum clip gap: 120 frames
- Confidence threshold: 0.8
- Fade duration: 15 frames

## Visualization Tools

The pipeline generates various visualizations:
- Training loss curves
- Performance metrics
- Learning rate schedules
- Confusion matrices
- Event distribution plots
- Score timelines
- Confidence distributions

## Error Handling

The pipeline includes robust error handling:
- Input validation
- Frame quality checks
- GPU memory management
- Automatic cleanup
- Detailed logging

## Monitoring and Logging

All components provide detailed logging:
- Processing statistics
- Error tracking
- Performance metrics
- Resource utilization

## Output Files

- `best_model.pth`: Best performing model weights
- `training_report.md`: Detailed training analysis
- `confusion_matrix.png`: Model performance visualization
- `processing_report.json`: Frame processing statistics
- `summary_report.json`: Overall analysis results
- `*_predictions.json`: Model predictions for each game

## Requirements
- Python 3.11+
- PyTorch with CUDA support (recommended)
- OpenCV
- NumPy
- Matplotlib
- PyQt5 (for annotation tool)
- scikit-learn
- seaborn

## Note
This implementation includes GPU optimizations but will fall back to CPU processing if CUDA is unavailable.

## Pipeline Workflow

The complete workflow should be executed in this order:

1. `preprocess.py` - Convert videos to frames
2. `annotate.py` - Manually annotate events
3. `create_labels.py` - Generate training labels
4. `train.py` - Train the model
5. `generate_predictions.py` - Generate predictions
6. `scoring.py` - Score and analyze predictions
7. `eval.py` - Evaluate model
8. `clip_generator.py` - Generate highlight clips