# Action Recognition System

A real-time action recognition system using MediaPipe Holistic and deep learning to detect and classify human actions from video input.

## Overview

This project implements an action recognition system that uses pose estimation and landmark detection to identify specific actions. The system leverages MediaPipe's Holistic model to extract body pose, hand, and face landmarks, which are then processed by an LSTM neural network for action classification.

## Features

- Real-time action detection from webcam feed
- MediaPipe Holistic integration for comprehensive body tracking
- LSTM-based deep learning model for action classification
- Support for multiple action classes: punch, kick, and "I love you" gesture
- Visual feedback with styled landmark rendering

## Actions Detected

- **Punch**: Detects punching motions
- **Kick**: Detects kicking motions
- **I Love You**: Detects the "I love you" sign language gesture

## Project Structure

```
action_recognition/
├── Dataset/                    # Training data organized by action
│   ├── punch/                 # Punch action sequences
│   ├── kick/                  # Kick action sequences
│   └── iloveyou/              # I love you gesture sequences
├── action_recogniser.ipynb    # Main notebook for action recognition
├── action_lstm.pth            # Trained LSTM model weights
├── action_lstm1.pth           # Alternative model weights
└── test.py                    # Testing script
```

## Requirements

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- TensorFlow/PyTorch
- Jupyter Notebook

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd action_recognition
```

2. Create and activate virtual environment:
```bash
python -m venv tfenv
tfenv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install opencv-python mediapipe numpy matplotlib tensorflow torch jupyter
```

## Usage

### Running the Jupyter Notebook

1. Activate the virtual environment
2. Launch Jupyter Notebook:
```bash
jupyter notebook
```
3. Open `action_recogniser.ipynb`
4. Run the cells to start real-time action recognition

### Key Functions

- `mediapipe_detection(image, model)`: Processes frames through MediaPipe
- `draw_landmarks(image, results)`: Draws basic landmarks on the frame
- `draw_styled_landmarks(image, results)`: Draws styled landmarks with custom colors

## Model Architecture

The system uses an LSTM (Long Short-Term Memory) neural network trained on sequences of body landmarks extracted from video frames. Each action is represented by 30 sequences of landmark data.

## Dataset Structure

Each action class contains 30 sequences (0-29), with each sequence representing a temporal series of frames capturing the action.

## How It Works

1. **Capture**: Video frames are captured from webcam or video file
2. **Detection**: MediaPipe Holistic extracts pose, hand, and face landmarks
3. **Feature Extraction**: Landmark coordinates are extracted and normalized
4. **Classification**: LSTM model processes the sequence and predicts the action
5. **Visualization**: Results are displayed with styled landmarks overlaid on the video

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe by Google for pose estimation
- YOLOv8 for alternative pose detection approach
