#!/usr/bin/env python3
"""
EEG LSL Streamer for Imagined Emotion Dataset

This script streams EEG data from the Imagined Emotion Dataset (OpenNeuro ds003004)
as an LSL stream, simulating a real-time EEG device.

Requirements:
    - pylsl
    - mne
    - numpy
    - pandas

Usage:
    python eeg_lsl_streamer.py --subject 1 --emotion joy
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import mne
from pylsl import StreamInfo, StreamOutlet, local_clock
import threading

class EEGLSLStreamer:
    def __init__(self, data_path, subject_num, emotion=None, stream_name="ImaginedEmotion", stream_type="EEG"):
        """
        Initialize EEG LSL Streamer
        
        Parameters:
            data_path (str): Path to the dataset directory
            subject_num (int): Subject number to stream (1-35, excluding 22)
            emotion (str, optional): Specific emotion to stream (e.g., 'joy', 'fear')
            stream_name (str): Name of the LSL stream
            stream_type (str): Type of the LSL stream
        """
        self.data_path = data_path
        self.subject_num = subject_num
        self.emotion = emotion
        
        # Format subject number with leading zero if needed
        self.subject_id = f"sub-{subject_num:02d}"
        self.stream_name = stream_name
        self.stream_type = stream_type
        
        # File paths
        self.eeg_file = os.path.join(data_path, self.subject_id, "eeg", 
                                     f"{self.subject_id}_task-ImaginedEmotion_eeg.set")
        self.events_file = os.path.join(data_path, self.subject_id, "eeg", 
                                        f"{self.subject_id}_task-ImaginedEmotion_events.tsv")
        
        # Check if files exist
        if not os.path.exists(self.eeg_file):
            raise FileNotFoundError(f"EEG file not found: {self.eeg_file}")
        if not os.path.exists(self.events_file):
            raise FileNotFoundError(f"Events file not found: {self.events_file}")
        
        # Load data
        self.load_data()
        
        # Initialize LSL stream
        self.initialize_lsl_stream()
        
        # Initialize marker stream for events
        self.initialize_marker_stream()
        
    def load_data(self):
        """Load EEG data and events"""
        print(f"Loading EEG data for subject {self.subject_id}...")
        self.raw = mne.io.read_raw_eeglab(self.eeg_file, preload=True)
        self.sfreq = self.raw.info['sfreq']
        self.ch_names = self.raw.ch_names
        
        print(f"Loading events from {self.events_file}...")
        self.events_df = pd.read_csv(self.events_file, sep='\t')
        
        # If specific emotion is requested, extract only those segments
        if self.emotion:
            self.extract_emotion_segments()
        else:
            # Get full data
            self.data, self.times = self.raw[:, :]
        
        print(f"Data loaded: {self.data.shape[1]} samples, {self.data.shape[0]} channels, {self.sfreq}Hz")
    
    def extract_emotion_segments(self):
        """Extract only segments corresponding to the specified emotion"""
        print(f"Extracting segments for emotion: {self.emotion}")
        
        # Find emotion onset and offset
        emotion_rows = self.events_df[self.events_df['value'] == self.emotion]
        if emotion_rows.empty:
            raise ValueError(f"No data for emotion '{self.emotion}' found in events file")
        
        # Find button presses that mark the start of experiencing the emotion
        emotion_onset = emotion_rows['onset'].values[0]
        
        # Find press1 events after the emotion onset
        press1_rows = self.events_df[(self.events_df['value'] == 'press1') & 
                                     (self.events_df['onset'] > emotion_onset)]
        
        if press1_rows.empty:
            raise ValueError(f"No press1 events found after emotion '{self.emotion}' onset")
            
        press1_onset = press1_rows['onset'].values[0]
        
        # Find exit event after press1
        exit_rows = self.events_df[(self.events_df['value'] == 'exit') & 
                                  (self.events_df['onset'] > press1_onset)]
        
        if exit_rows.empty:
            # If no exit event found, use the next emotion onset or end of recording
            next_emotion_rows = self.events_df[(self.events_df['onset'] > press1_onset) & 
                                              (self.events_df['value'].isin(['awe', 'joy', 'love', 'compassion', 
                                                                             'happy', 'nurturant', 'desire', 
                                                                             'anger', 'fear', 'sad', 'grief', 
                                                                             'disgust', 'frustration', 'guilty', 'amusement']))]
            if next_emotion_rows.empty:
                emotion_offset = self.times[-1]
            else:
                emotion_offset = next_emotion_rows['onset'].values[0]
        else:
            emotion_offset = exit_rows['onset'].values[0]
        
        # Convert times to samples
        onset_sample = int(press1_onset * self.sfreq)
        offset_sample = int(emotion_offset * self.sfreq)
        
        # Extract data segment
        self.data, self.times = self.raw[:, onset_sample:offset_sample]
        
        # Create new events dataframe containing only relevant events
        self.emotion_events_df = self.events_df[(self.events_df['onset'] >= press1_onset) & 
                                               (self.events_df['onset'] <= emotion_offset)].copy()
        
        # Adjust onset times to be relative to the new segment
        self.emotion_events_df['onset'] = self.emotion_events_df['onset'] - press1_onset
        
        print(f"Extracted {self.emotion} segment: {self.data.shape[1]} samples ({self.data.shape[1]/self.sfreq:.2f} seconds)")
    
    def initialize_lsl_stream(self):
        """Initialize LSL stream for EEG data"""
        # Create a new stream info
        info = StreamInfo(
            name=self.stream_name,
            type=self.stream_type,
            channel_count=len(self.ch_names),
            nominal_srate=self.sfreq,
            channel_format='float32',
            source_id=f'imaginedemotion_{self.subject_id}'
        )
        
        # Add channel information
        channels = info.desc().append_child("channels")
        for ch_name in self.ch_names:
            channels.append_child("channel") \
                .append_child_value("label", ch_name) \
                .append_child_value("unit", "microvolts") \
                .append_child_value("type", "EEG")
        
        # Create outlet
        self.outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)
        print(f"LSL EEG outlet created: {self.stream_name}")
    
    def initialize_marker_stream(self):
        """Initialize LSL stream for event markers"""
        # Create a new stream info for markers
        marker_info = StreamInfo(
            name=f"{self.stream_name}_markers",
            type='Markers',
            channel_count=1,
            nominal_srate=0,  # Irregular sample rate
            channel_format='string',
            source_id=f'imaginedemotion_{self.subject_id}_markers'
        )
        
        # Create outlet for markers
        self.marker_outlet = StreamOutlet(marker_info)
        print(f"LSL marker outlet created: {self.stream_name}_markers")
    
    def stream_data(self, chunk_size=32, speedup=1.0):
        """
        Stream EEG data through LSL
        
        Parameters:
            chunk_size (int): Number of samples to send in each chunk
            speedup (float): Speed multiplier (1.0 = real-time, 2.0 = twice as fast)
        """
        print(f"Starting to stream data for {self.subject_id} (speedup={speedup}x)...")
        
        # Calculate the sleep duration for real-time playback
        sleep_duration = chunk_size / (self.sfreq * speedup)
        
        # Create a lookup of event onsets in samples
        if hasattr(self, 'emotion_events_df'):
            events_df = self.emotion_events_df
        else:
            events_df = self.events_df
        
        event_samples = {int(row['onset'] * self.sfreq): row['value'] 
                         for _, row in events_df.iterrows()}
        
        try:
            # Stream data in chunks
            for i in range(0, self.data.shape[1], chunk_size):
                # Get current chunk (ensure we don't go out of bounds)
                end_idx = min(i + chunk_size, self.data.shape[1])
                chunk = self.data[:, i:end_idx]
                
                # Check if there are any events in this chunk
                for j in range(i, end_idx):
                    if j in event_samples:
                        # Send marker
                        marker_value = str(event_samples[j])
                        self.marker_outlet.push_sample([marker_value])
                        print(f"Marker sent: {marker_value} at sample {j}")
                
                # Stream EEG chunk
                self.outlet.push_chunk(chunk.T.tolist())
                
                # Sleep to maintain real-time rate
                time.sleep(sleep_duration)
                
                # Print progress occasionally
                if i % (chunk_size * 100) == 0:
                    print(f"Streamed {i/self.sfreq:.1f}s / {self.data.shape[1]/self.sfreq:.1f}s")
                    
            print(f"Finished streaming {self.data.shape[1]/self.sfreq:.1f} seconds of data")
            
        except KeyboardInterrupt:
            print("Streaming interrupted by user")
    
    def stream_continuous(self, chunk_size=32, speedup=1.0):
        """
        Stream EEG data continuously in a separate thread
        
        Parameters:
            chunk_size (int): Number of samples to send in each chunk
            speedup (float): Speed multiplier (1.0 = real-time, 2.0 = twice as fast)
        """
        self.streaming_thread = threading.Thread(
            target=self.stream_data,
            kwargs={'chunk_size': chunk_size, 'speedup': speedup}
        )
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        return self.streaming_thread


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stream EEG data from Imagined Emotion Dataset via LSL')
    
    parser.add_argument('--data_path', type=str, 
                        default='/home/vihaanthat/Documents/sem6/BTP/imaginedEmotionDataset/ds003004',
                        help='Path to the dataset directory')
    
    parser.add_argument('--subject', type=int, required=True,
                        help='Subject number to stream (1-35, excluding 22)')
    
    parser.add_argument('--emotion', type=str, 
                        choices=['awe', 'joy', 'love', 'compassion', 'happy', 'nurturant', 
                                'desire', 'anger', 'fear', 'sad', 'grief', 'disgust', 
                                'frustration', 'guilty', 'amusement'],
                        help='Specific emotion to stream (optional)')
    
    parser.add_argument('--chunk_size', type=int, default=32,
                        help='Number of samples to send in each chunk')
    
    parser.add_argument('--speedup', type=float, default=1.0,
                        help='Speed multiplier (1.0 = real-time, 2.0 = twice as fast)')
    
    parser.add_argument('--stream_name', type=str, default='ImaginedEmotion',
                        help='Name of the LSL stream')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Create and start streamer
    streamer = EEGLSLStreamer(
        data_path=args.data_path,
        subject_num=args.subject,
        emotion=args.emotion,
        stream_name=args.stream_name
    )
    
    # Stream data (this will block until finished)
    streamer.stream_data(chunk_size=args.chunk_size, speedup=args.speedup)
