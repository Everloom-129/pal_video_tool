# pal_video_tool/src/__init__.py

from .api_client import APIClient
from .video_processor import VideoProcessor
from .visualization import Visualizer

__version__ = "0.1.0"
__author__ = "Tony Wang"

def greet():
    print("Welcome to use pal_video_tool!")