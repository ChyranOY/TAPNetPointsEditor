#!/usr/bin/env python3
"""
TAPNet Tracker - A comprehensive video tracking solution using TAPNext model.

This package provides:
- Video preprocessing and tracking
- Interactive editing capabilities
- Visualization tools
- Gradio-based user interface
"""

__version__ = "1.0.0"
__author__ = "TAPNet Tracker Team"

from .core.tracker import TAPNextTracker
from .ui.gradio_app import create_tapnext_app

__all__ = [
    "TAPNextTracker",
    "create_tapnext_app",
] 