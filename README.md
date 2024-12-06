# Baseball Game Analysis Pipeline

This document outlines the complete pipeline for analyzing baseball game videos to detect and score key events.

## Setup
In the directory that has the source code:
```
brew install python@3.11
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Directory Structure

```
baseball-analysis/
├── training_videos/           # Input video files
├── processed_frames/     # Extracted and processed video frames
├── checkpoints/         # Saved model weights
├── game_scores/        # Individual game scoring files
└── logs/               # Pipeline execution logs
```

## Pipeline Steps

### 1. Video Preprocessing (`preprocess.py`)

**Purpose**: Convert raw game footage into processed frames suitable for analysis.

**Input**:
- Raw MP4 video files in `training_videos/`

**Output**:
- Processed frame files in `processed_frames/`
  - Size: 640x480 pixels
  - Format: Normalized numpy arrays (.npy) and thumbnails (.jpg)
  - Sampling: 1 frame every 10 frames

**Key Parameters**:
- `target_size`: (640, 480)
- `sample_rate`: 10
- `max_frames`: 15000 per video

### 2. Label Creation (`create_labels.py`)

**Purpose**: Generate labeled data for training the event detection model.

**Output**:
- `train_labels.json`: Training set labels
- `val_labels.json`: Validation set labels

**Event Types**:
1. Home Run (event_type: 1)
2. Out (event_type: 2)
3. Hit (event_type: 3)
4. No Event (event_type: 0)

### 3. Model Training (`train.py`)

**Purpose**: Train the 3D CNN model and generate event predictions.

**Input**:
- Processed frames from `processed_frames/`
- Labels from `train_labels.json` and `val_labels.json`

**Output**:
- Trained model: `checkpoints/best_model.pth`
- Predictions: `predictions.json`

**Key Parameters**:
- Learning rate: 0.0001
- Batch size: 4
- Epochs: 20
- Sequence length: 32 frames

### 4. Game Scoring (`scoring.py`)

**Purpose**: Apply rule-based scoring to detected events.

**Input**:
- Event predictions from `predictions.json`

**Output**:
- Individual game scores in `game_scores/`
- Overall `scoring_report.json`

**Scoring Rules**:
1. Home Run:
   - Base score: 4.0
   - Confidence multiplier: 1.2
   - Consecutive multiplier: 1.5
   - Late-game multiplier: 1.3

2. Out:
   - Base score: 2.0
   - Confidence multiplier: 1.1
   - Consecutive multiplier: 1.2
   - Late-game multiplier: 1.2

3. Hit:
   - Base score: 1.0
   - Confidence multiplier: 1.1
   - Consecutive multiplier: 1.2
   - Late-game multiplier: 1.2

## Error Handling

The pipeline includes checks for:
- Empty input directories
- Missing intermediate files
- Failed processing steps

If an error occurs, check:
1. The log file for detailed error messages
2. Input video format and content
3. Available disk space
4. GPU memory (for training)

## Output Verification

After pipeline completion, verify:
1. Processed frames exist in `processed_frames/`
2. Model file exists in `checkpoints/`
3. Prediction and scoring files are generated
4. Log file shows successful completion