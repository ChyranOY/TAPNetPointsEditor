#!/usr/bin/env python3
"""
Video processing utilities for TAPNet Tracker.

Contains functions for video preprocessing, frame extraction, and format conversion.
"""

import cv2
import numpy as np
from typing import Tuple


def extract_first_frame(video_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Extract first frame from video for point selection, return frame and size info in original dimensions"""
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Unable to read first frame from video")
    
    # Get original dimensions
    original_height, original_width = frame.shape[:2]
    original_size = (original_width, original_height)  # (width, height)
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cap.release()
    return frame_rgb, original_size


def preprocess_video(video_path: str, target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Preprocess video file, return both original video data and size information"""
    cap = cv2.VideoCapture(video_path)
    
    # Get original video information
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_size = (original_width, original_height)  # (width, height)
    
    frames_processed = []  # 256x256 frames for inference
    frames_original = []   # Original size frames for visualization
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save original size frames
        frames_original.append(frame_rgb.copy())
        
        # Resize to 256x256 for inference
        frame_resized = cv2.resize(frame_rgb, target_size)
        
        # Normalize to [-1, 1]
        frame_normalized = (frame_resized.astype(np.float32) / 255.0) * 2.0 - 1.0
        
        frames_processed.append(frame_normalized)
    
    cap.release()
    
    if not frames_processed:
        raise ValueError("Unable to read frames from video")
    
    frame_count = len(frames_processed)
    print(f"Video processing completed:")
    print(f"  - Frame count: {frame_count}")
    print(f"  - Original size: {original_size} (width x height)")
    print(f"  - Inference size: {target_size} (width x height)")
    
    return (np.stack(frames_processed, axis=0), 
            np.stack(frames_original, axis=0), 
            original_size) 