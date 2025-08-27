"""
User interface module for TAPNet Tracker.

Contains the Gradio-based web interface.
"""

from .gradio_app import create_tapnext_app

__all__ = [
    "create_tapnext_app",
] 