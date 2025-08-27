#!/usr/bin/env python3
"""
Configuration module for TAPNet Tracker.

Contains default configurations and settings.
"""

import os
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for TAPNet Tracker."""
    
    # Model configuration
    MODEL_WIDTH: int = 768
    MODEL_NUM_HEADS: int = 12
    MODEL_KERNEL_SIZE: int = 4
    MODEL_NUM_BLOCKS: int = 12
    
    # Default model checkpoint path
    DEFAULT_CHECKPOINT_PATH: str = "/path/to/tapnet/bootstapnext_ckpt.npz"
    
    # Video processing configuration
    TARGET_SIZE: Tuple[int, int] = (256, 256)
    QUANT_MULTI: int = 8
    
    # Visualization configuration
    DEFAULT_FPS: int = 15
    TRACKS_LEAVE_TRACE: int = 16
    MAX_DISTANCE_FOR_POINT_SELECTION: int = 20
    
    # Output configuration - 使用绝对路径确保文件保存在同一位置
    DEFAULT_OUTPUT_DIR: str = "/path/to/outputs"
    DEFAULT_NUM_POINTS: int = 32
    DEFAULT_POINT_METHOD: str = "manual"
    
    # UI configuration
    MAX_VIDEO_DISPLAY_SIZE: Tuple[int, int] = (512, 512)
    
    @classmethod
    def get_checkpoint_path(cls, custom_path: str = None) -> str:
        """Get checkpoint path, use custom path if provided and exists."""
        if custom_path and os.path.exists(custom_path):
            return custom_path
        return cls.DEFAULT_CHECKPOINT_PATH
    
    @classmethod
    def ensure_output_dir(cls, output_dir: str = None) -> str:
        """Ensure output directory exists and return the path."""
        dir_path = output_dir or cls.DEFAULT_OUTPUT_DIR
        os.makedirs(dir_path, exist_ok=True)
        return dir_path


# Global configuration instance
config = Config() 