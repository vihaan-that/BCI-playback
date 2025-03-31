# BCI-Playback: 

A tool for streaming EEG data from the OpenNeuro Imagined Emotion Dataset (ds003004) over Lab Streaming Layer (LSL). This enables simulation of real-time EEG data for BCI application development and testing.

## Overview

This repository contains tools to:

1. Load EEG recordings of imagined emotions from the OpenNeuro dataset
2. Stream the EEG data in real-time via LSL (Lab Streaming Layer)
3. Transmit event markers that indicate emotion onsets and button presses
4. Receive and process the streaming EEG data for machine learning applications

## Dataset Information

The [Imagined Emotion Dataset (ds003004)](https://openneuro.org/datasets/ds003004/versions/1.1.1) contains high-density EEG recordings of participants imagining 15 different emotional states:

- **Positive emotions**: joy, happiness, awe, love, compassion
- **Negative emotions**: sadness, anger, fear, frustration, disgust
- **Study design**: Participants listened to voice recordings suggesting scenarios to imagine experiencing specific target emotions
- **Data collection**: When participants began to feel the target emotion, they indicated this by pressing a button
- **Data characteristics**: 224 EEG channels, 256 Hz sampling rate, preprocessed with 1-Hz high-pass filter

## Installation

### Prerequisites

- Python 3.8+
- git-annex (for downloading dataset files)
- DataLad (for managing dataset)

### Setup

1. **Clone this repository**:
   ```bash
   git clone https://github.com/vihaan-that/BCI-playback.git
   cd BCI-playback
   ```

2. **Install dependencies**:
   ```bash
   # Install required Python packages
   pip install -r requirements.txt
   
   # Install liblsl library (required by pylsl)
   conda install -c conda-forge liblsl
   # Or on Debian/Ubuntu:
   # sudo apt-get install liblsl
   ```

3. **Download the dataset**:
   ```bash
   # Install DataLad if not already installed
   pip install datalad
   
   # Install git-annex if not already installed
   # On Debian/Ubuntu:
   sudo apt-get install git-annex
   
   # Download the dataset
   datalad install https://github.com/OpenNeuroDatasets/ds003004.git
   
   # Enter the dataset directory
   cd ds003004
   
   # Download data for a specific subject (e.g., subject 1)
   datalad get sub-01/eeg/sub-01_task-ImaginedEmotion_eeg.set
   datalad get sub-01/eeg/sub-01_task-ImaginedEmotion_eeg.fdt
   ```

## Usage

### Streaming EEG Data

The `eeg_lsl_streamer.py` script streams EEG data from the dataset via LSL:

```bash
# Stream all data from subject 1 at real-time speed
python eeg_lsl_streamer.py --subject 1 --data_path /path/to/ds003004

# Stream only "joy" emotion data from subject 1
python eeg_lsl_streamer.py --subject 1 --emotion joy --data_path /path/to/ds003004

# Stream data at 2x speed
python eeg_lsl_streamer.py --subject 1 --speedup 2.0 --data_path /path/to/ds003004
```

### Receiving EEG Data

The `lsl_receiver_example.py` script demonstrates how to receive and process the streamed EEG data:

```bash
python lsl_receiver_example.py
```

### Command-line Arguments

#### For `eeg_lsl_streamer.py`:

- `--data_path`: Path to the dataset directory (default: current directory)
- `--subject`: Subject number to stream (1-35, excluding 22)
- `--emotion`: Specific emotion to stream (optional, choices: awe, joy, love, compassion, happy, anger, fear, sad, disgust, frustration)
- `--chunk_size`: Number of samples to send in each chunk (default: 32)
- `--speedup`: Speed multiplier (1.0 = real-time, 2.0 = twice as fast)
- `--stream_name`: Name of the LSL stream (default: "ImaginedEmotion")

## Examples

### Basic Streaming

```bash
# Make sure you're in the BCI-playback directory
cd BCI-playback

# Stream data from subject 1
python eeg_lsl_streamer.py --subject 1 --data_path /path/to/ds003004
```

### Machine Learning Integration Example

Here's an example of how you might integrate this with a machine learning pipeline:

1. Start the streamer in one terminal:
   ```bash
   python eeg_lsl_streamer.py --subject 1 --emotion joy --data_path /path/to/ds003004
   ```

2. In your ML application, use code similar to:
   ```python
   from pylsl import StreamInlet, resolve_streams
   import numpy as np
   from sklearn.preprocessing import StandardScaler
   
   # Connect to LSL stream
   streams = resolve_streams()
   eeg_streams = [s for s in streams if s.name() == "ImaginedEmotion"]
   inlet = StreamInlet(eeg_streams[0])
   
   # Process data in real-time
   while True:
       chunk, timestamps = inlet.pull_chunk()
       if chunk:
           # Extract features from the chunk (e.g., spectral power)
           features = extract_features(chunk)
           
           # Make predictions with your ML model
           prediction = model.predict(features)
           
           # Take action based on prediction
           print(f"Detected emotion: {prediction}")
   ```

## Troubleshooting

### Common Issues

- **LSL library not found**: Install the liblsl library with `conda install -c conda-forge liblsl`
- **Dataset files missing**: Use DataLad to download the specific files: `datalad get path/to/file`
- **No streams found**: Make sure the streamer is running first before starting the receiver

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset from [OpenNeuro ds003004](https://openneuro.org/datasets/ds003004/versions/1.1.1)
- References:
  - Onton J.A. and Makeig S. (2009). High-frequency broadband modulations of electroencephalographic spectra. Front. Hum. Neurosci.
  - Kothe C.A., Makeig S., and Onton J.A. (2013). Emotion Recognition from EEG during Self-Paced Emotional Imagery.
  - Hsu S.H., et al. (2022). Unsupervised learning of brain state dynamics during emotion imagination using high-density EEG.
# BCI-playback
This repository is built to playback a pre-recorded EEG data stream to compensate for the lack of sophisticated BCI headsets in my experimentation.
