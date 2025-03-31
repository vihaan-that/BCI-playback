#!/usr/bin/env python3
"""
EEG LSL Receiver Example

This script receives EEG data from an LSL stream and prints basic information.
It can be used to verify that the EEG LSL Streamer is working correctly.

Requirements:
    - pylsl
    - numpy
    - matplotlib (optional, for visualization)
"""

import time
import numpy as np
from pylsl import StreamInlet, resolve_streams
import threading

def receive_eeg_stream(stream_name="ImaginedEmotion", timeout=10):
    """
    Receive and process EEG data from an LSL stream
    
    Parameters:
        stream_name (str): Name of the LSL stream to look for
        timeout (float): Time to wait for stream in seconds
    """
    print(f"Looking for an EEG stream named '{stream_name}'...")
    # Using the correct approach to find streams by name
    streams = resolve_streams(timeout)
    
    # Filter to find the stream with the matching name
    eeg_streams = [s for s in streams if s.name() == stream_name]
    
    if not eeg_streams:
        print(f"No stream found with name '{stream_name}'. Available streams:")
        for s in streams:
            print(f"  - {s.name()} ({s.type()}, {s.channel_count()} channels)")
        return
    
    # Create an inlet to receive from the first stream
    inlet = StreamInlet(eeg_streams[0])
    print(f"Connected to '{stream_name}' stream with {inlet.info().channel_count()} channels")
    
    # Get stream information
    info = inlet.info()
    sampling_rate = info.nominal_srate()
    channel_count = info.channel_count()
    
    # Get channel names
    ch_names = []
    ch = info.desc().child("channels").child("channel")
    for i in range(channel_count):
        ch_names.append(ch.child_value("label"))
        ch = ch.next_sibling()
    
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"First few channels: {ch_names[:5]}...")
    
    # Process data
    try:
        # Buffer for visualization (10 seconds of data)
        buffer_size = int(sampling_rate * 10)
        data_buffer = np.zeros((channel_count, buffer_size))
        
        print("Receiving data...")
        sample_count = 0
        start_time = time.time()
        
        while True:
            # Get a new sample (or chunk of samples)
            chunk, timestamps = inlet.pull_chunk()
            
            if timestamps:
                # Count samples and update the buffer
                chunk_size = len(timestamps)
                sample_count += chunk_size
                
                # Show the latest signal value from the first few channels
                latest_values = [chunk[-1][i] for i in range(min(5, channel_count))]
                
                # Calculate streaming rate
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = sample_count / elapsed
                    
                    # Print status every second
                    if sample_count % int(sampling_rate) < chunk_size:
                        print(f"Received {sample_count} samples ({rate:.1f} Hz). Latest values: {latest_values}")
            
            # Small sleep to reduce CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nReceiver stopped by user")


def receive_marker_stream(stream_name="ImaginedEmotion_markers", timeout=10):
    """
    Receive and process marker data from an LSL stream
    
    Parameters:
        stream_name (str): Name of the LSL marker stream to look for
        timeout (float): Time to wait for stream in seconds
    """
    print(f"Looking for a marker stream named '{stream_name}'...")
    # Using the correct approach to find streams by name
    streams = resolve_streams(timeout)
    
    # Filter to find the stream with the matching name
    marker_streams = [s for s in streams if s.name() == stream_name]
    
    if not marker_streams:
        print(f"No marker stream found with name '{stream_name}'")
        return
    
    # Create an inlet to receive from the marker stream
    inlet = StreamInlet(marker_streams[0])
    print(f"Connected to '{stream_name}' stream")
    
    # Process markers
    try:
        print("Receiving markers...")
        
        while True:
            # Get a new marker (timeout ensures we don't block forever)
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            
            if sample:
                print(f"Marker received: {sample[0]} at time {timestamp:.3f}")
            
            # Small sleep to reduce CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nMarker receiver stopped by user")


if __name__ == "__main__":
    # Start marker receiver in a separate thread
    marker_thread = threading.Thread(target=receive_marker_stream)
    marker_thread.daemon = True
    marker_thread.start()
    
    # Run EEG receiver in the main thread
    receive_eeg_stream()
